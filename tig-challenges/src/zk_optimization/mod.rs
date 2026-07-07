pub mod baselines;
pub(crate) mod crypto;
pub(crate) mod dag;
pub(crate) mod r1cs;

use crate::QUALITY_PRECISION;
pub use crypto::CryptoHash;
pub use dag::{CircuitConfig, DAG};
pub use r1cs::{R1CSMatrix, SpartanInstance, WitnessError};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

pub use curve25519_dalek::scalar::Scalar;
use libspartan::{InputsAssignment, Instance, SNARKGens, VarsAssignment, SNARK};
use merlin::Transcript;

use dag::generate_dag;
use r1cs::{compute_witness, dag_to_spartan, satisfies, solve_witness_forward};

// =============================================================================
// Data Structures
// =============================================================================

impl_kv_string_serde! {
    Track {
        delta: usize,
    }
}

impl_base64_serde! {
    Solution {
        circuit_star: SpartanInstance,
        y0_pub: Vec<Scalar>,
        y_star_pub: Vec<Scalar>,
        proof0: SNARK,
        proof_star: SNARK,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub delta: usize,
    pub circuit_c0: SpartanInstance,
    pub num_circuit_inputs: usize,
    pub num_circuit_outputs: usize,
}

// =============================================================================
// Solver Interface
// =============================================================================

/// Callback signature for the participant's circuit optimizer.
///
/// Takes the baseline circuit C⁰ (as a [`SpartanInstance`]) and returns C* —
/// an optimized circuit with strictly fewer constraints that computes the same
/// function.
///
/// # Requirements
///
/// ## 1. Fewer constraints
/// `C*.num_cons` must be strictly less than `C⁰.num_cons`.
///
/// ## 2. Functional equivalence
/// C* must compute the same function as C⁰ at the evaluation point `x_eval`
/// derived from `H(C⁰) || H(C*)`.
///
/// ## 3. Topological evaluation order (CRITICAL)
/// Constraint rows must be ordered so that when the witness solver processes
/// row `i`, all variables in A and B are already known from previous rows or
/// from the circuit inputs. At most one variable may be unknown per row.
///
/// Violations are rejected immediately with `NotInEvaluationOrder { row }`.
/// C⁰ is already in topological order — optimizers that only remove or merge
/// constraints naturally preserve it.
pub type OptimizeCircuitFn = fn(&SpartanInstance) -> SpartanInstance;

// =============================================================================
// Helpers
// =============================================================================

fn seed_to_hex(seed: &[u8; 32]) -> String {
    seed.iter().map(|b| format!("{:02x}", b)).collect()
}

fn num_non_zero(si: &SpartanInstance) -> usize {
    si.A.len().max(si.B.len()).max(si.C.len())
}

// =============================================================================
// Challenge Implementation
// =============================================================================

impl Challenge {
    /// Generates a challenge instance from a seed and track.
    ///
    /// `seed → hex → SHA256 → ChaCha20 → backward-BFS DAG → R1CS (C⁰)`
    pub fn generate_instance(seed: &[u8; 32], track: &Track) -> Result<Challenge> {
        let seed_hex = seed_to_hex(seed);
        let config = CircuitConfig::from_delta(track.delta);
        let dag = generate_dag(&seed_hex, &config);
        let circuit_c0 = dag_to_spartan(&dag);

        Ok(Challenge {
            seed: *seed,
            delta: track.delta.clone(),
            num_circuit_inputs: dag.num_inputs,
            num_circuit_outputs: dag.num_outputs,
            circuit_c0,
        })
    }

    /// Produces a Solution for the challenge using the participant's optimizer.
    ///
    /// 1. Derives `x_eval = H(H(C⁰) || H(C*))` (anti-grinding)
    /// 2. Computes C⁰ witness via DAG (`compute_witness`)
    /// 3. Computes C* witness via single forward pass (`solve_witness_forward`)
    /// 4. Generates Spartan proofs π⁰ and π*
    pub fn build_solution(&self, circuit_star: &SpartanInstance) -> Result<Solution> {
        // 1. Derive x_eval
        let h0 = CryptoHash::from_serializable(&self.circuit_c0)?;
        let h_star = CryptoHash::from_serializable(circuit_star)?;
        let x_eval = h0.combine(&h_star).to_scalars(self.num_circuit_inputs);

        // 2. C⁰ witness (regenerate DAG)
        let seed_hex = seed_to_hex(&self.seed);
        let config = CircuitConfig::from_delta(self.delta);
        let dag = generate_dag(&seed_hex, &config);
        let (vars0, public_io0) = compute_witness(&dag, &x_eval);

        // 3. C* witness (single forward pass — rows must be in topological order)
        let (vars_star, public_io_star) =
            solve_witness_forward(circuit_star, self.num_circuit_outputs, &x_eval)
                .map_err(|e| anyhow!("C* witness solver failed: {:?}", e))?;

        // 4a. Prove π⁰
        let c0 = &self.circuit_c0;
        let inst0 = Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C)
            .map_err(|e| anyhow!("Spartan instance for C⁰: {:?}", e))?;
        let gens0 = SNARKGens::new(c0.num_cons, c0.num_vars, c0.num_inputs, num_non_zero(c0));
        let (comm0, decomm0) = SNARK::encode(&inst0, &gens0);

        let av0 = VarsAssignment::new(&scalars_to_bytes(&vars0))
            .map_err(|e| anyhow!("VarsAssignment for C⁰: {:?}", e))?;
        let ai0 = InputsAssignment::new(&scalars_to_bytes(&public_io0))
            .map_err(|e| anyhow!("InputsAssignment for C⁰: {:?}", e))?;

        let proof0 = SNARK::prove(
            &inst0,
            &comm0,
            &decomm0,
            av0,
            &ai0,
            &gens0,
            &mut Transcript::new(b"ZKChallenge_C0"),
        );

        // 4b. Prove π*
        let inst_star = Instance::new(
            circuit_star.num_cons,
            circuit_star.num_vars,
            circuit_star.num_inputs,
            &circuit_star.A,
            &circuit_star.B,
            &circuit_star.C,
        )
        .map_err(|e| anyhow!("Spartan instance for C*: {:?}", e))?;
        let gens_star = SNARKGens::new(
            circuit_star.num_cons,
            circuit_star.num_vars,
            circuit_star.num_inputs,
            num_non_zero(&circuit_star),
        );
        let (comm_star, decomm_star) = SNARK::encode(&inst_star, &gens_star);

        let av_star = VarsAssignment::new(&scalars_to_bytes(&vars_star))
            .map_err(|e| anyhow!("VarsAssignment for C*: {:?}", e))?;
        let ai_star = InputsAssignment::new(&scalars_to_bytes(&public_io_star))
            .map_err(|e| anyhow!("InputsAssignment for C*: {:?}", e))?;

        let proof_star = SNARK::prove(
            &inst_star,
            &comm_star,
            &decomm_star,
            av_star,
            &ai_star,
            &gens_star,
            &mut Transcript::new(b"ZKChallenge_Cstar"),
        );

        let y0_pub = public_io0[..self.num_circuit_outputs].to_vec();
        let y_star_pub = public_io_star[..self.num_circuit_outputs].to_vec();

        Ok(Solution {
            circuit_star: circuit_star.clone(),
            y0_pub,
            y_star_pub,
            proof0,
            proof_star,
        })
    }

    /// Verifies a solution against this challenge.
    ///
    /// 1. Recomputes `x_eval = H(H(C⁰) || H(C*))`
    /// 2. Checks `K* < K⁰`
    /// 3. Checks `y⁰_pub == y*_pub`
    /// 4. Recomputes Spartan parameters for C⁰; verifies π⁰
    /// 5. Recomputes Spartan parameters for C*; verifies π*
    conditional_pub!(
        fn evaluate_num_constraints(&self, solution: &Solution) -> Result<usize> {
            // 1. Derive evaluation point
            let h0 = CryptoHash::from_serializable(&self.circuit_c0)?;
            let h_star = CryptoHash::from_serializable(&solution.circuit_star)?;
            let x_eval = h0.combine(&h_star).to_scalars(self.num_circuit_inputs);

            // 2. Constraint reduction
            if solution.circuit_star.num_cons >= self.circuit_c0.num_cons {
                return Err(anyhow!(
                    "C* has {} constraints, must be < {} (C⁰)",
                    solution.circuit_star.num_cons,
                    self.circuit_c0.num_cons
                ));
            }

            // 2a. C* must preserve C0's public I/O shape — a sound optimization
            // cannot change the function's arity. This also bounds num_inputs so
            // the witness allocation in solve_witness_forward cannot OOM.
            if solution.circuit_star.num_inputs != self.circuit_c0.num_inputs
                || solution.circuit_star.num_outputs != self.num_circuit_outputs
            {
                return Err(anyhow!(
                    "C* I/O shape ({}/{}) differs from C0 ({}/{})",
                    solution.circuit_star.num_inputs,
                    solution.circuit_star.num_outputs,
                    self.circuit_c0.num_inputs,
                    self.num_circuit_outputs,
                ));
            }

            // 2b. C* must be a well-defined FUNCTION of its inputs, not merely a
            // satisfiable relation. Otherwise π* only proves that *some* witness
            // exists at x_eval, and a prover could "optimize" by deleting
            // constraints — the original witness still satisfies the reduced
            // circuit, so π* and the y0 == y* check would both pass even though
            // C* no longer computes C⁰ (Schwartz-Zippel does not apply to a
            // relation). The hardened forward pass verifies C* is
            // triangularizable; re-checking the solved witness against every row
            // makes "C* is a function" self-contained (consistent ∧ triangular
            // ⇒ unique witness per input).
            let (vars_star, io_star_solved) =
                solve_witness_forward(&solution.circuit_star, self.num_circuit_outputs, &x_eval)
                    .map_err(|e| anyhow!("C* is not a well-formed function: {:?}", e))?;
            if !satisfies(&solution.circuit_star, &vars_star, &io_star_solved) {
                return Err(anyhow!(
                    "C* forward-solved witness does not satisfy all constraints"
                ));
            }
            // Bind C*'s real, forward-solved outputs directly to the claimed
            // y_star_pub, so output-correctness does not rest on π* alone. A
            // malicious prover could otherwise set y_star_pub ≠ what C* actually
            // computes while π* attests only that the (wrong) public I/O is
            // consistent with some witness. Combined with step 3 (y0_pub ==
            // y_star_pub) this pins C*'s output to C0's.
            let solved_outputs = io_star_solved.get(..self.num_circuit_outputs);
            if solved_outputs.map_or(true, |o| o != solution.y_star_pub.as_slice()) {
                return Err(anyhow!("C* solved outputs != claimed y_star_pub"));
            }

            // 3. Output equivalence
            if solution.y0_pub != solution.y_star_pub {
                return Err(anyhow!(
                    "Output mismatch: C⁰ and C* produced different outputs"
                ));
            }

            // 4. Verify π⁰
            let c0 = &self.circuit_c0;
            let inst0 = Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C)
                .map_err(|e| anyhow!("Spartan instance for C⁰: {:?}", e))?;
            let gens0 = SNARKGens::new(c0.num_cons, c0.num_vars, c0.num_inputs, num_non_zero(c0));
            let (comm0, _) = SNARK::encode(&inst0, &gens0);

            let mut io0: Vec<Scalar> = solution.y0_pub.clone();
            io0.extend_from_slice(&x_eval);
            let io0_bytes: Vec<[u8; 32]> = io0.iter().map(|s| s.to_bytes()).collect();
            let assignment_io0 = InputsAssignment::new(&io0_bytes)
                .map_err(|e| anyhow!("InputsAssignment for C⁰: {:?}", e))?;

            solution
                .proof0
                .verify(
                    &comm0,
                    &assignment_io0,
                    &mut Transcript::new(b"ZKChallenge_C0"),
                    &gens0,
                )
                .map_err(|e| anyhow!("π⁰ verification failed: {:?}", e))?;

            // 5. Verify π*
            let cs = &solution.circuit_star;
            let inst_star =
                Instance::new(cs.num_cons, cs.num_vars, cs.num_inputs, &cs.A, &cs.B, &cs.C)
                    .map_err(|e| anyhow!("Spartan instance for C*: {:?}", e))?;
            let gens_star =
                SNARKGens::new(cs.num_cons, cs.num_vars, cs.num_inputs, num_non_zero(cs));
            let (comm_star, _) = SNARK::encode(&inst_star, &gens_star);

            let mut io_star: Vec<Scalar> = solution.y_star_pub.clone();
            io_star.extend_from_slice(&x_eval);
            let io_star_bytes: Vec<[u8; 32]> = io_star.iter().map(|s| s.to_bytes()).collect();
            let assignment_io_star = InputsAssignment::new(&io_star_bytes)
                .map_err(|e| anyhow!("InputsAssignment for C*: {:?}", e))?;

            solution
                .proof_star
                .verify(
                    &comm_star,
                    &assignment_io_star,
                    &mut Transcript::new(b"ZKChallenge_Cstar"),
                    &gens_star,
                )
                .map_err(|e| anyhow!("π* verification failed: {:?}", e))?;

            Ok(solution.circuit_star.num_cons)
        }
    );

    conditional_pub!(
        fn compute_baseline(&self) -> Result<usize> {
            Ok(baselines::remove_aliases(&self.circuit_c0).num_cons)
        }
    );

    conditional_pub!(
        fn evaluate_solution(&self, solution: &Solution) -> Result<i32> {
            let num_constraints = self.evaluate_num_constraints(solution)?;
            let baseline_num_constraints = self.compute_baseline()?;
            let quality = (baseline_num_constraints as f64 - num_constraints as f64)
                / (baseline_num_constraints as f64);
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
        }
    );
}

fn scalars_to_bytes(scalars: &[Scalar]) -> Vec<[u8; 32]> {
    scalars.iter().map(|s| s.to_bytes()).collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn make_challenge(delta: usize) -> Challenge {
        let mut seed = [0u8; 32];
        seed[0] = 42;
        Challenge::generate_instance(&seed, &Track { delta }).unwrap()
    }

    #[test]
    fn test_generate_instance() {
        let t0 = Instant::now();
        let ch = make_challenge(1);
        eprintln!(
            "[generate_instance] delta=1 -> {} cons, {} vars, {} pub_io in {:.2?}",
            ch.circuit_c0.num_cons,
            ch.circuit_c0.num_vars,
            ch.circuit_c0.num_inputs,
            t0.elapsed()
        );
        assert!(ch.circuit_c0.num_cons >= 900);
        assert!(ch.circuit_c0.num_vars > 0);
        assert!(ch.circuit_c0.num_inputs > 0);
    }

    #[test]
    fn test_deterministic_generation() {
        let c1 = make_challenge(1);
        let c2 = make_challenge(1);
        assert_eq!(c1.circuit_c0.num_cons, c2.circuit_c0.num_cons);
        assert_eq!(c1.circuit_c0.A, c2.circuit_c0.A);
        eprintln!(
            "[deterministic] OK ({} constraints)",
            c1.circuit_c0.num_cons
        );
    }

    #[test]
    fn test_hash_and_xeval_derivation() {
        let ch = make_challenge(1);
        let h0 = CryptoHash::from_serializable(&ch.circuit_c0).unwrap();
        let h0_again = CryptoHash::from_serializable(&ch.circuit_c0).unwrap();
        assert_eq!(h0, h0_again);

        let x_eval = h0.combine(&h0_again).to_scalars(ch.num_circuit_inputs);
        assert_eq!(x_eval.len(), ch.num_circuit_inputs);
        for (i, s) in x_eval.iter().enumerate() {
            assert_ne!(*s, Scalar::ZERO, "x_eval[{}] should not be zero", i);
        }
    }

    #[test]
    fn test_witness_satisfies_c0() {
        let ch = make_challenge(1);
        let c0 = &ch.circuit_c0;
        let h0 = CryptoHash::from_serializable(c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

        let seed_hex = seed_to_hex(&ch.seed);
        let config = CircuitConfig::from_delta(ch.delta);
        let dag_val = generate_dag(&seed_hex, &config);
        let (vars, public_io) = compute_witness(&dag_val, &x_eval);

        let inst =
            Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C).unwrap();
        let av = VarsAssignment::new(&scalars_to_bytes(&vars)).unwrap();
        let ai = InputsAssignment::new(&scalars_to_bytes(&public_io)).unwrap();
        assert!(inst.is_sat(&av, &ai).unwrap(), "Witness must satisfy C⁰");
    }

    #[test]
    fn test_full_identity_roundtrip() {
        let total = Instant::now();
        let ch = make_challenge(1);
        let c0 = &ch.circuit_c0;
        eprintln!("\n=== Identity Roundtrip (delta=1) ===");
        eprintln!(
            "[1] {} cons, {} vars, {} pub_io",
            c0.num_cons, c0.num_vars, c0.num_inputs
        );

        let h0 = CryptoHash::from_serializable(c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

        let seed_hex = seed_to_hex(&ch.seed);
        let config = CircuitConfig::from_delta(ch.delta);
        let dag_val = generate_dag(&seed_hex, &config);
        let (vars0, io0) = compute_witness(&dag_val, &x_eval);

        let inst0 =
            Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C).unwrap();
        let gens0 = SNARKGens::new(c0.num_cons, c0.num_vars, c0.num_inputs, num_non_zero(c0));
        let (comm0, decomm0) = SNARK::encode(&inst0, &gens0);
        let av0 = VarsAssignment::new(&scalars_to_bytes(&vars0)).unwrap();
        let ai0 = InputsAssignment::new(&scalars_to_bytes(&io0)).unwrap();

        let t0 = Instant::now();
        let proof0 = SNARK::prove(
            &inst0,
            &comm0,
            &decomm0,
            av0,
            &ai0,
            &gens0,
            &mut Transcript::new(b"ZKChallenge_C0"),
        );
        eprintln!("[2] prove pi0 in {:.2?}", t0.elapsed());

        let t0 = Instant::now();
        proof0
            .verify(
                &comm0,
                &ai0,
                &mut Transcript::new(b"ZKChallenge_C0"),
                &gens0,
            )
            .expect("pi0 must verify");
        eprintln!("[3] verify pi0 in {:.2?}", t0.elapsed());

        // C* = C0 (identity), witness via forward pass
        let (vars_star, io_star) =
            solve_witness_forward(c0, ch.num_circuit_outputs, &x_eval).unwrap();
        let gens_star = SNARKGens::new(c0.num_cons, c0.num_vars, c0.num_inputs, num_non_zero(c0));
        let (comm_star, decomm_star) = SNARK::encode(&inst0, &gens_star);
        let av_star = VarsAssignment::new(&scalars_to_bytes(&vars_star)).unwrap();
        let ai_star = InputsAssignment::new(&scalars_to_bytes(&io_star)).unwrap();

        let t0 = Instant::now();
        let proof_star = SNARK::prove(
            &inst0,
            &comm_star,
            &decomm_star,
            av_star,
            &ai_star,
            &gens_star,
            &mut Transcript::new(b"ZKChallenge_Cstar"),
        );
        eprintln!("[4] prove pi* in {:.2?}", t0.elapsed());

        let t0 = Instant::now();
        proof_star
            .verify(
                &comm_star,
                &ai_star,
                &mut Transcript::new(b"ZKChallenge_Cstar"),
                &gens_star,
            )
            .expect("pi* must verify");
        eprintln!("[5] verify pi* in {:.2?}", t0.elapsed());

        assert_eq!(
            &io0[..ch.num_circuit_outputs],
            &io_star[..ch.num_circuit_outputs]
        );
        eprintln!("=== PASSED in {:.2?} ===\n", total.elapsed());
    }

    #[test]
    fn test_alias_optimizer_roundtrip() {
        let total = Instant::now();
        eprintln!("\n=== Alias Optimizer Roundtrip (delta=1) ===");

        let ch = make_challenge(1);
        eprintln!("[1] baseline: {} constraints", ch.circuit_c0.num_cons);

        fn alias_optimizer(c0: &SpartanInstance) -> SpartanInstance {
            baselines::remove_aliases(c0)
        }

        let t0 = Instant::now();
        let circuit_star = alias_optimizer(&ch.circuit_c0);
        let solution = ch
            .build_solution(&circuit_star)
            .expect("build_solution must succeed");
        eprintln!(
            "[2] solve: {} -> {} constraints in {:.2?}",
            ch.circuit_c0.num_cons,
            solution.circuit_star.num_cons,
            t0.elapsed()
        );

        assert!(solution.circuit_star.num_cons < ch.circuit_c0.num_cons);

        let t0 = Instant::now();
        ch.evaluate_num_constraints(&solution)
            .expect("evaluate_num_constraints must succeed");
        eprintln!("[3] verify OK in {:.2?}", t0.elapsed());

        let eps = 1.0 - solution.circuit_star.num_cons as f64 / ch.circuit_c0.num_cons as f64;
        eprintln!("[4] epsilon = {:.4}", eps);
        eprintln!("=== PASSED in {:.2?} ===\n", total.elapsed());
    }

    #[test]
    fn test_wrong_order_rejected() {
        let ch = make_challenge(1);
        let c0 = &ch.circuit_c0;
        let num_cons = c0.num_cons;

        let flip = |mat: &R1CSMatrix| -> R1CSMatrix {
            mat.iter()
                .map(|&(row, col, val)| (num_cons - 1 - row, col, val))
                .collect()
        };

        let reversed = SpartanInstance {
            num_cons: c0.num_cons,
            num_vars: c0.num_vars,
            num_inputs: c0.num_inputs,
            num_outputs: c0.num_outputs,
            A: flip(&c0.A),
            B: flip(&c0.B),
            C: flip(&c0.C),
        };

        let h0 = CryptoHash::from_serializable(c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

        let result = solve_witness_forward(&reversed, ch.num_circuit_outputs, &x_eval);
        assert!(
            matches!(result, Err(WitnessError::NotInEvaluationOrder { .. })),
            "Expected NotInEvaluationOrder, got: {:?}",
            result
        );
        eprintln!(
            "[wrong_order] correctly rejected: {:?}",
            result.unwrap_err()
        );
    }

    #[test]
    fn test_nonlinear_row_rejected() {
        // A row where the unknown appears in BOTH A and B is quadratic in that
        // variable and does not pin a unique value — the circuit is not a
        // function. The hardened forward solver must reject it with
        // NonLinearUnknown instead of silently mis-solving it.
        //
        // Constraint: (z0 + 5) * (z0 + 3) = z0
        //   z layout: [z0 (private) | 1 (const, col 1) | input0 (col 2)]
        let inst = SpartanInstance {
            num_cons: 1,
            num_vars: 1,
            num_inputs: 1,
            num_outputs: 0,
            A: vec![
                (0, 0, Scalar::ONE.to_bytes()),
                (0, 1, Scalar::from(5u64).to_bytes()),
            ],
            B: vec![
                (0, 0, Scalar::ONE.to_bytes()),
                (0, 1, Scalar::from(3u64).to_bytes()),
            ],
            C: vec![(0, 0, Scalar::ONE.to_bytes())],
        };
        let x_eval = vec![Scalar::from(7u64)];
        let res = solve_witness_forward(&inst, 0, &x_eval);
        assert!(
            matches!(res, Err(WitnessError::NonLinearUnknown { row: 0 })),
            "quadratic defining row must be rejected, got: {:?}",
            res
        );
    }

    #[test]
    fn test_cheat_deleted_rows_rejected() {
        // Attack from the soundness analysis: a malicious prover "optimizes" by
        // simply deleting constraints from C⁰. The reduced circuit is still
        // satisfiable by the original witness (so π* would verify and y0 == y*
        // would hold), but it is no longer a function — the variables whose
        // defining rows were removed are left unconstrained. The hardened
        // forward solver must reject it.
        let ch = make_challenge(1);
        let c0 = &ch.circuit_c0;

        let drop_every = 3usize; // keep 2 of every 3 rows
        let mut remap: Vec<Option<usize>> = vec![None; c0.num_cons];
        let mut new_row = 0usize;
        for old in 0..c0.num_cons {
            if old % drop_every == 0 {
                continue;
            }
            remap[old] = Some(new_row);
            new_row += 1;
        }
        if new_row >= c0.num_cons {
            return; // nothing was deleted; vacuous
        }
        let pick = |m: &R1CSMatrix| -> R1CSMatrix {
            m.iter()
                .filter_map(|&(r, col, val)| remap[r].map(|nr| (nr, col, val)))
                .collect()
        };
        let cheat = SpartanInstance {
            num_cons: new_row,
            num_vars: c0.num_vars,
            num_inputs: c0.num_inputs,
            num_outputs: c0.num_outputs,
            A: pick(&c0.A),
            B: pick(&c0.B),
            C: pick(&c0.C),
        };

        let h0 = CryptoHash::from_serializable(c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);
        let res = solve_witness_forward(&cheat, ch.num_circuit_outputs, &x_eval);
        assert!(
            res.is_err(),
            "delete-rows cheat must be rejected (C* would not be a function), got: {:?}",
            res
        );
        eprintln!("[cheat] deleted-rows C* correctly rejected: {:?}", res);
    }

    /// Minimal structurally-valid instance for the malformed-input tests below:
    /// 1 constraint, 1 private var, 1 input, 0 outputs.
    /// z layout: `[z0 (col 0) | 1 (col 1) | input0 (col 2)]`, z_len = 3.
    fn minimal_instance() -> SpartanInstance {
        SpartanInstance {
            num_cons: 1,
            num_vars: 1,
            num_inputs: 1,
            num_outputs: 0,
            A: vec![(0, 0, Scalar::ONE.to_bytes())],
            B: vec![(0, 1, Scalar::ONE.to_bytes())],
            C: vec![(0, 2, Scalar::ONE.to_bytes())],
        }
    }

    #[test]
    fn test_malformed_non_canonical_rejected() {
        // A matrix scalar whose little-endian value exceeds the Curve25519
        // scalar modulus is rejected by from_canonical_bytes (returns None).
        // validate_instance must surface this as MalformedInstance rather than
        // letting the build_row_views `.unwrap()` panic.
        let mut inst = minimal_instance();
        let mut bad = Scalar::ONE.to_bytes();
        bad[31] = 0xff; // 0xff in the top byte ≫ field modulus ⇒ non-canonical
        inst.A = vec![(0, 0, bad)];

        let x_eval = vec![Scalar::from(7u64)];
        let res = solve_witness_forward(&inst, 0, &x_eval);
        assert!(
            matches!(res, Err(WitnessError::MalformedInstance { .. })),
            "non-canonical scalar must be rejected, got: {:?}",
            res
        );
    }

    #[test]
    fn test_malformed_row_out_of_range_rejected() {
        // num_cons = 1 ⇒ only row 0 exists; an entry claiming row 5 must be
        // rejected before build_row_views indexes a_rows[5] (out of bounds).
        let mut inst = minimal_instance();
        inst.A = vec![(5, 0, Scalar::ONE.to_bytes())];

        let x_eval = vec![Scalar::from(7u64)];
        let res = solve_witness_forward(&inst, 0, &x_eval);
        assert!(
            matches!(res, Err(WitnessError::MalformedInstance { .. })),
            "out-of-range row must be rejected, got: {:?}",
            res
        );
    }

    #[test]
    fn test_malformed_col_out_of_range_rejected() {
        // z_len = 3 ⇒ valid columns are 0..3; col 7 would index past the
        // z-vector in accumulate/dot.
        let mut inst = minimal_instance();
        inst.A = vec![(0, 7, Scalar::ONE.to_bytes())];

        let x_eval = vec![Scalar::from(7u64)];
        let res = solve_witness_forward(&inst, 0, &x_eval);
        assert!(
            matches!(res, Err(WitnessError::MalformedInstance { .. })),
            "out-of-range column must be rejected, got: {:?}",
            res
        );
    }

    #[test]
    fn test_malformed_huge_num_vars_rejected() {
        // Absurd num_vars with tiny num_cons: validate_instance catches
        // num_vars > num_cons (a precondition for any triangularizable circuit)
        // BEFORE the z allocation — no panic, no hang, no giant Vec.
        let mut inst = minimal_instance();
        inst.num_vars = 1usize << 50;
        inst.num_cons = 1;

        let x_eval = vec![Scalar::from(7u64)];
        let res = solve_witness_forward(&inst, 0, &x_eval);
        assert!(
            matches!(res, Err(WitnessError::MalformedInstance { .. })),
            "huge num_vars must be rejected without panicking, got: {:?}",
            res
        );
    }

    #[test]
    fn test_f5_wrong_outputs_rejected() {
        // F5 defense-in-depth: the verifier binds C*'s forward-solved outputs
        // directly to the claimed y_star_pub (step 2b), instead of relying on
        // π* (verified only in step 5). A malicious prover could lie about
        // outputs *consistently* — y0_pub == y_star_pub == W, while C* really
        // computes R ≠ W. Step 3 (y0_pub == y_star_pub) would then pass; without
        // the F5 check, only π (checked later) would catch it. The F5 check
        // rejects it in step 2b, before any proof is verified.
        let ch = make_challenge(1);
        let circuit_star = baselines::remove_aliases(&ch.circuit_c0);
        let mut solution = ch.build_solution(&circuit_star).expect("valid solution");

        assert!(
            !solution.y_star_pub.is_empty(),
            "test requires at least one circuit output"
        );
        // Claim a wrong output consistently in BOTH public fields, so step 3
        // (y0_pub == y_star_pub) does not catch it.
        let wrong = solution.y_star_pub[0] + Scalar::ONE;
        if let Some(o) = solution.y0_pub.first_mut() {
            *o = wrong;
        }
        if let Some(o) = solution.y_star_pub.first_mut() {
            *o = wrong;
        }

        let res = ch.evaluate_num_constraints(&solution);
        let err = res.expect_err("tampered outputs must be rejected by the F5 check");
        assert!(
            format!("{:#}", err).contains("solved outputs"),
            "expected the F5 solved-outputs check to fire, got: {:?}",
            err
        );
    }
}
