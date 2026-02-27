use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use curve25519_dalek::scalar::Scalar;

use libspartan::{InputsAssignment, Instance, SNARKGens, VarsAssignment, SNARK};

use blake3::Hasher as Blake3;
use merlin::Transcript;
use tig_circuit_tools::{
    compute_witness, dag_to_spartan, generate_dag, solve_witness_from_r1cs, CircuitConfig,
    SpartanInstance,
};

// =============================================================================
// Utility: Cryptographic Hashing and Hash-to-Field
// =============================================================================

/// Sparse R1CS matrix in COO (Coordinate) format.
pub type R1CSMatrix = Vec<(usize, usize, [u8; 32])>;

/// Cryptographic hash (512-bit) used for anti-grinding commitments.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct CryptoHash(#[serde(with = "serde_bytes")] pub [u8; 64]);

impl CryptoHash {
    /// Hashes a Circuit structure: H(C).
    pub fn from_circuit(circuit: &Circuit) -> Result<Self> {
        let bytes = bincode::serialize(circuit)?;
        let mut hasher = Blake3::new();
        hasher.update(&bytes);
        let mut hash = [0u8; 64];
        hasher.finalize_xof().fill(&mut hash);
        Ok(CryptoHash(hash))
    }

    /// Combines two hashes: H(H1 || H2).
    pub fn combine(&self, other: &CryptoHash) -> Self {
        let mut hasher = Blake3::new();
        hasher.update(&self.0);
        hasher.update(&other.0);
        let mut hash = [0u8; 64];
        hasher.finalize_xof().fill(&mut hash);
        CryptoHash(hash)
    }

    /// Hash-to-Field: derives `count` Scalars from this hash digest.
    /// Implements r_i = H(seed || i) mod P.
    pub fn to_scalars(&self, count: usize) -> Vec<Scalar> {
        (0..count)
            .map(|i| {
                let mut hasher = Blake3::new();
                hasher.update(&self.0);
                hasher.update(&(i as u64).to_le_bytes());
                let mut buf = [0u8; 64];
                hasher.finalize_xof().fill(&mut buf);
                Scalar::from_bytes_mod_order_wide(&buf)
            })
            .collect()
    }
}

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Difficulty {
    pub delta: usize,
}

impl From<Vec<i32>> for Difficulty {
    fn from(arr: Vec<i32>) -> Self {
        Self {
            delta: arr[0] as usize,
        }
    }
}

impl From<Difficulty> for Vec<i32> {
    fn from(d: Difficulty) -> Vec<i32> {
        vec![d.delta as i32]
    }
}

/// R1CS circuit representation compatible with libspartan.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[allow(non_snake_case)]
pub struct Circuit {
    pub num_cons: usize,
    pub num_vars: usize,
    pub num_inputs: usize,
    pub A: R1CSMatrix,
    pub B: R1CSMatrix,
    pub C: R1CSMatrix,
}

impl Circuit {
    /// Constructs a local Circuit from a tig-circuit-tools SpartanInstance.
    fn from_spartan_instance(si: tig_circuit_tools::SpartanInstance) -> Self {
        Circuit {
            num_cons: si.num_cons,
            num_vars: si.num_vars,
            num_inputs: si.num_inputs,
            A: si.A,
            B: si.B,
            C: si.C,
        }
    }

    fn num_non_zero(&self) -> usize {
        self.A.len().max(self.B.len()).max(self.C.len())
    }

    fn to_spartan_instance(&self) -> SpartanInstance {
        SpartanInstance {
            num_cons: self.num_cons,
            num_vars: self.num_vars,
            num_inputs: self.num_inputs,
            A: self.A.clone(),
            B: self.B.clone(),
            C: self.C.clone(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub circuit_c0: Circuit,
    pub num_circuit_inputs: usize,
    pub num_circuit_outputs: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Solution {
    pub circuit_star: Circuit,
    pub y0_pub: Vec<Scalar>,
    pub y_star_pub: Vec<Scalar>,
    pub proof0: SNARK,
    pub proof_star: SNARK,
}

// =============================================================================
// Solver Interface
// =============================================================================

/// Callback signature for the participant's circuit optimizer.
///
/// Takes the baseline circuit C⁰ and returns C* — an optimized circuit with
/// fewer constraints. Witness generation is handled automatically by
/// `solve_witness_from_r1cs` (fixed-point R1CS solver).
pub type OptimizeCircuitFn = fn(&Circuit) -> Circuit;

// =============================================================================
// Helper: seed bytes to hex string
// =============================================================================

fn seed_to_hex(seed: &[u8; 32]) -> String {
    seed.iter().map(|b| format!("{:02x}", b)).collect()
}

// =============================================================================
// Challenge Implementation
// =============================================================================

impl Challenge {
    /// Generates a challenge instance from a seed and difficulty.
    ///
    /// 1. Converts the seed to a hex string for DAG generation.
    /// 2. Generates a random DAG via tig-circuit-tools.
    /// 3. Converts the DAG to a Spartan R1CS instance (the baseline circuit C⁰).
    pub fn generate_instance(seed: &[u8; 32], difficulty: &Difficulty) -> Result<Challenge> {
        let seed_hex = seed_to_hex(seed);
        let config = CircuitConfig::from_difficulty(difficulty.delta as u32);
        let dag = generate_dag(&seed_hex, &config);
        let spartan_inst = dag_to_spartan(&dag);

        let num_circuit_inputs = dag.num_inputs;
        let num_circuit_outputs = dag.num_outputs;
        let circuit_c0 = Circuit::from_spartan_instance(spartan_inst);

        Ok(Challenge {
            seed: *seed,
            difficulty: difficulty.clone(),
            circuit_c0,
            num_circuit_inputs,
            num_circuit_outputs,
        })
    }

    /// Verifies a solution against this challenge.
    ///
    /// The verifier recomputes generators and commitments from the circuit
    /// matrices — it does not trust any prover-supplied values beyond the
    /// proofs, outputs, and optimized circuit.
    ///
    /// Checks:
    /// 1. K* < K⁰ (optimized circuit has fewer constraints)
    /// 2. y⁰_pub == y*_pub (both circuits produce the same outputs)
    /// 3. π⁰ verifies against C⁰ with public I/O [y0_pub..., x_eval...]
    /// 4. π* verifies against C* with public I/O [y_star_pub..., x_eval...]
    pub fn verify_solution(&self, solution: &Solution) -> Result<()> {
        // 1. Compute hashes and derive evaluation point
        let h0 = CryptoHash::from_circuit(&self.circuit_c0)?;
        let h_star = CryptoHash::from_circuit(&solution.circuit_star)?;
        let combined_hash = h0.combine(&h_star);
        let x_eval = combined_hash.to_scalars(self.num_circuit_inputs);

        // 2. Constraint reduction check: K* < K⁰
        if solution.circuit_star.num_cons >= self.circuit_c0.num_cons {
            return Err(anyhow!(
                "Optimized circuit has {} constraints, must be < {} (baseline)",
                solution.circuit_star.num_cons,
                self.circuit_c0.num_cons
            ));
        }

        // 3. Output equivalence: y⁰_pub == y*_pub
        if solution.y0_pub != solution.y_star_pub {
            return Err(anyhow!(
                "Output mismatch: C⁰ and C* produced different outputs"
            ));
        }

        // 4. Recompute Spartan instance, generators, and commitment for C⁰
        let c0 = &self.circuit_c0;
        let inst0 = Instance::new(
            c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C,
        )
        .map_err(|e| anyhow!("Failed to create Spartan Instance for C⁰: {:?}", e))?;
        let gens0 = SNARKGens::new(
            c0.num_cons, c0.num_vars, c0.num_inputs, c0.num_non_zero(),
        );
        let (comm0, _) = SNARK::encode(&inst0, &gens0);

        // 5. Verify π⁰
        let mut io0: Vec<Scalar> = solution.y0_pub.clone();
        io0.extend_from_slice(&x_eval);
        let io0_bytes: Vec<[u8; 32]> = io0.iter().map(|s| s.to_bytes()).collect();
        let assignment_io0 = InputsAssignment::new(&io0_bytes)
            .map_err(|e| anyhow!("Failed to create InputsAssignment for C⁰: {:?}", e))?;

        let mut transcript0 = Transcript::new(b"ZKChallenge_C0");
        solution
            .proof0
            .verify(&comm0, &assignment_io0, &mut transcript0, &gens0)
            .map_err(|e| anyhow!("Proof π⁰ verification failed: {:?}", e))?;

        // 6. Recompute Spartan instance, generators, and commitment for C*
        let c_star = &solution.circuit_star;
        let inst_star = Instance::new(
            c_star.num_cons, c_star.num_vars, c_star.num_inputs,
            &c_star.A, &c_star.B, &c_star.C,
        )
        .map_err(|e| anyhow!("Failed to create Spartan Instance for C*: {:?}", e))?;
        let gens_star = SNARKGens::new(
            c_star.num_cons, c_star.num_vars, c_star.num_inputs, c_star.num_non_zero(),
        );
        let (comm_star, _) = SNARK::encode(&inst_star, &gens_star);

        // 7. Verify π*
        let mut io_star: Vec<Scalar> = solution.y_star_pub.clone();
        io_star.extend_from_slice(&x_eval);
        let io_star_bytes: Vec<[u8; 32]> = io_star.iter().map(|s| s.to_bytes()).collect();
        let assignment_io_star = InputsAssignment::new(&io_star_bytes)
            .map_err(|e| anyhow!("Failed to create InputsAssignment for C*: {:?}", e))?;

        let mut transcript_star = Transcript::new(b"ZKChallenge_Cstar");
        solution
            .proof_star
            .verify(&comm_star, &assignment_io_star, &mut transcript_star, &gens_star)
            .map_err(|e| anyhow!("Proof π* verification failed: {:?}", e))?;

        Ok(())
    }
}

// =============================================================================
// Solver
// =============================================================================

/// Generates a solution for the given challenge using the participant's optimizer.
///
/// 1. Calls `optimize` to produce C*.
/// 2. Derives x_eval from H(C⁰) || H(C*) (anti-grinding).
/// 3. Regenerates the DAG and computes the C⁰ witness via `compute_witness`.
/// 4. Computes the C* witness via `solve_witness_from_r1cs` (fixed-point R1CS solver).
/// 5. Generates Spartan SNARK proofs π⁰ and π*.
pub fn solve_challenge(challenge: &Challenge, optimize: OptimizeCircuitFn) -> Result<Solution> {
    // 1. Participant optimizes C⁰ → C*
    let circuit_star = optimize(&challenge.circuit_c0);

    // 2. Compute hashes
    let h0 = CryptoHash::from_circuit(&challenge.circuit_c0)?;
    let h_star = CryptoHash::from_circuit(&circuit_star)?;

    // 3. Derive evaluation point x_eval = Hash-to-Field(H(C⁰) || H(C*))
    let combined_hash = h0.combine(&h_star);
    let x_eval = combined_hash.to_scalars(challenge.num_circuit_inputs);

    // 4. Compute witness for C⁰ by regenerating the DAG
    let seed_hex = seed_to_hex(&challenge.seed);
    let config = CircuitConfig::from_difficulty(challenge.difficulty.delta as u32);
    let dag = generate_dag(&seed_hex, &config);
    let (vars0, public_io0) = compute_witness(&dag, &x_eval);
    // public_io0 = [y0_out_0, ..., y0_out_n, x_eval_0, ..., x_eval_m]

    // 5. Compute witness for C* using fixed-point R1CS solver
    let (vars_star, public_io_star) = solve_witness_from_r1cs(
        &circuit_star.to_spartan_instance(),
        challenge.num_circuit_outputs,
        &x_eval,
    )
    .map_err(|e| anyhow!("C* witness solver failed: {:?}", e))?;

    // 6. Generate Spartan proof for C⁰
    let inst0 = Instance::new(
        challenge.circuit_c0.num_cons,
        challenge.circuit_c0.num_vars,
        challenge.circuit_c0.num_inputs,
        &challenge.circuit_c0.A,
        &challenge.circuit_c0.B,
        &challenge.circuit_c0.C,
    )
    .map_err(|e| anyhow!("Failed to create Spartan Instance for C⁰: {:?}", e))?;

    let gens0 = SNARKGens::new(
        challenge.circuit_c0.num_cons,
        challenge.circuit_c0.num_vars,
        challenge.circuit_c0.num_inputs,
        challenge.circuit_c0.num_non_zero(),
    );

    let (comm0, decomm0) = SNARK::encode(&inst0, &gens0);

    let vars0_bytes: Vec<[u8; 32]> = vars0.iter().map(|s| s.to_bytes()).collect();
    let io0_bytes: Vec<[u8; 32]> = public_io0.iter().map(|s| s.to_bytes()).collect();
    let assignment_vars0 = VarsAssignment::new(&vars0_bytes)
        .map_err(|e| anyhow!("Failed to create VarsAssignment for C⁰: {:?}", e))?;
    let assignment_io0 = InputsAssignment::new(&io0_bytes)
        .map_err(|e| anyhow!("Failed to create InputsAssignment for C⁰: {:?}", e))?;

    let mut transcript0 = Transcript::new(b"ZKChallenge_C0");
    let proof0 = SNARK::prove(
        &inst0,
        &comm0,
        &decomm0,
        assignment_vars0,
        &assignment_io0,
        &gens0,
        &mut transcript0,
    );

    // 7. Generate Spartan proof for C*
    let inst_star = Instance::new(
        circuit_star.num_cons,
        circuit_star.num_vars,
        circuit_star.num_inputs,
        &circuit_star.A,
        &circuit_star.B,
        &circuit_star.C,
    )
    .map_err(|e| anyhow!("Failed to create Spartan Instance for C*: {:?}", e))?;

    let gens_star = SNARKGens::new(
        circuit_star.num_cons,
        circuit_star.num_vars,
        circuit_star.num_inputs,
        circuit_star.num_non_zero(),
    );

    let (comm_star, decomm_star) = SNARK::encode(&inst_star, &gens_star);

    let vars_star_bytes: Vec<[u8; 32]> = vars_star.iter().map(|s| s.to_bytes()).collect();
    let io_star_bytes: Vec<[u8; 32]> = public_io_star.iter().map(|s| s.to_bytes()).collect();
    let assignment_vars_star = VarsAssignment::new(&vars_star_bytes)
        .map_err(|e| anyhow!("Failed to create VarsAssignment for C*: {:?}", e))?;
    let assignment_io_star = InputsAssignment::new(&io_star_bytes)
        .map_err(|e| anyhow!("Failed to create InputsAssignment for C*: {:?}", e))?;

    let mut transcript_star = Transcript::new(b"ZKChallenge_Cstar");
    let proof_star = SNARK::prove(
        &inst_star,
        &comm_star,
        &decomm_star,
        assignment_vars_star,
        &assignment_io_star,
        &gens_star,
        &mut transcript_star,
    );

    // 8. Extract output portions (first num_circuit_outputs elements of public_io)
    let y0_pub = public_io0[..challenge.num_circuit_outputs].to_vec();
    let y_star_pub = public_io_star[..challenge.num_circuit_outputs].to_vec();

    Ok(Solution {
        circuit_star,
        y0_pub,
        y_star_pub,
        proof0,
        proof_star,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use tig_circuit_tools::remove_aliases;

    /// Helper: builds a Challenge with a fixed test seed.
    fn make_challenge(delta: usize) -> Challenge {
        let mut seed = [0u8; 32];
        seed[0] = 42;
        let difficulty = Difficulty { delta };
        Challenge::generate_instance(&seed, &difficulty).unwrap()
    }

    // ----- Unit-level tests (fast, no Spartan proofs) -----

    #[test]
    fn test_generate_instance() {
        let t0 = Instant::now();
        let ch = make_challenge(1);
        let elapsed = t0.elapsed();
        eprintln!("[generate_instance] delta=1 -> {} constraints, {} vars, {} public I/O ({} inputs, {} outputs) in {:.2?}",
            ch.circuit_c0.num_cons, ch.circuit_c0.num_vars, ch.circuit_c0.num_inputs,
            ch.num_circuit_inputs, ch.num_circuit_outputs, elapsed);
        assert!(ch.circuit_c0.num_cons >= 900, "delta=1 should give ~1000 constraints");
        assert!(ch.circuit_c0.num_vars > 0);
        assert!(ch.circuit_c0.num_inputs > 0);
        assert!(ch.num_circuit_inputs > 0);
        assert!(ch.num_circuit_outputs > 0);
    }

    #[test]
    fn test_deterministic_generation() {
        let c1 = make_challenge(1);
        let c2 = make_challenge(1);
        assert_eq!(c1.circuit_c0.num_cons, c2.circuit_c0.num_cons);
        assert_eq!(c1.circuit_c0.num_vars, c2.circuit_c0.num_vars);
        assert_eq!(c1.circuit_c0.num_inputs, c2.circuit_c0.num_inputs);
        assert_eq!(c1.circuit_c0.A, c2.circuit_c0.A);
        eprintln!("[deterministic] OK: same seed -> identical circuit ({} constraints)", c1.circuit_c0.num_cons);
    }

    #[test]
    fn test_hash_and_xeval_derivation() {
        let ch = make_challenge(1);
        let t0 = Instant::now();
        let h0 = CryptoHash::from_circuit(&ch.circuit_c0).unwrap();
        let hash_time = t0.elapsed();
        let h0_again = CryptoHash::from_circuit(&ch.circuit_c0).unwrap();
        assert_eq!(h0, h0_again, "Hashing must be deterministic");

        let combined = h0.combine(&h0_again);
        let x_eval = combined.to_scalars(ch.num_circuit_inputs);
        assert_eq!(x_eval.len(), ch.num_circuit_inputs);

        for (i, s) in x_eval.iter().enumerate() {
            assert_ne!(*s, Scalar::ZERO, "x_eval[{}] should not be zero", i);
        }
        eprintln!("[hash+xeval] Blake3 hash of circuit in {:.2?}, derived {} x_eval scalars", hash_time, x_eval.len());
    }

    #[test]
    fn test_witness_satisfies_c0() {
        let ch = make_challenge(1);
        let c0 = &ch.circuit_c0;

        let h0 = CryptoHash::from_circuit(c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

        let t0 = Instant::now();
        let seed_hex = seed_to_hex(&ch.seed);
        let config = CircuitConfig::from_difficulty(ch.difficulty.delta as u32);
        let dag = generate_dag(&seed_hex, &config);
        let (vars, public_io) = compute_witness(&dag, &x_eval);
        let witness_time = t0.elapsed();

        assert_eq!(vars.len(), c0.num_vars, "vars length must match num_vars");
        assert_eq!(public_io.len(), c0.num_inputs, "public_io length must match num_inputs");

        let inst = Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C)
            .unwrap();
        let vars_bytes: Vec<[u8; 32]> = vars.iter().map(|s| s.to_bytes()).collect();
        let io_bytes: Vec<[u8; 32]> = public_io.iter().map(|s| s.to_bytes()).collect();
        let av = VarsAssignment::new(&vars_bytes).unwrap();
        let ai = InputsAssignment::new(&io_bytes).unwrap();

        let t1 = Instant::now();
        assert!(inst.is_sat(&av, &ai).unwrap(), "Witness must satisfy C⁰");
        let sat_time = t1.elapsed();

        eprintln!("[witness] DAG regen + witness computation in {:.2?}", witness_time);
        eprintln!("[witness] is_sat check in {:.2?}", sat_time);
        eprintln!("[witness] {} private vars, {} public I/O values", vars.len(), public_io.len());
    }

    // ----- Integration test: full Spartan prove/verify round-trip -----

    #[test]
    fn test_full_identity_roundtrip() {
        let total = Instant::now();

        // --- Step 1: generate challenge ---
        let t0 = Instant::now();
        let ch = make_challenge(1);
        let c0 = &ch.circuit_c0;
        eprintln!("\n=== Full Identity Roundtrip (delta=1) ===");
        eprintln!("[1/7] generate_instance: {} constraints, {} vars, {} public I/O ({} in, {} out), nnz=({},{},{}) in {:.2?}",
            c0.num_cons, c0.num_vars, c0.num_inputs,
            ch.num_circuit_inputs, ch.num_circuit_outputs,
            c0.A.len(), c0.B.len(), c0.C.len(),
            t0.elapsed());

        // --- Step 2: hash derivation & x_eval ---
        let t0 = Instant::now();
        let h0 = CryptoHash::from_circuit(c0).unwrap();
        let h_star = h0.clone();
        let x_eval = h0.combine(&h_star).to_scalars(ch.num_circuit_inputs);
        eprintln!("[2/7] hash + x_eval derivation ({} scalars) in {:.2?}", x_eval.len(), t0.elapsed());

        // --- Step 3: compute witness for C⁰ ---
        let t0 = Instant::now();
        let seed_hex = seed_to_hex(&ch.seed);
        let config = CircuitConfig::from_difficulty(ch.difficulty.delta as u32);
        let dag = generate_dag(&seed_hex, &config);
        let (vars0, public_io0) = compute_witness(&dag, &x_eval);
        eprintln!("[3/7] DAG regen + witness computation in {:.2?}", t0.elapsed());

        // --- Step 4: encode + prove π⁰ ---
        let t0 = Instant::now();
        let inst0 = Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C)
            .unwrap();
        let gens0 = SNARKGens::new(c0.num_cons, c0.num_vars, c0.num_inputs, c0.num_non_zero());
        let (comm0, decomm0) = SNARK::encode(&inst0, &gens0);
        let encode_time = t0.elapsed();

        let vars0_bytes: Vec<[u8; 32]> = vars0.iter().map(|s| s.to_bytes()).collect();
        let io0_bytes: Vec<[u8; 32]> = public_io0.iter().map(|s| s.to_bytes()).collect();
        let av0 = VarsAssignment::new(&vars0_bytes).unwrap();
        let ai0 = InputsAssignment::new(&io0_bytes).unwrap();

        let t0 = Instant::now();
        let mut t_prove0 = Transcript::new(b"ZKChallenge_C0");
        let proof0 = SNARK::prove(&inst0, &comm0, &decomm0, av0, &ai0, &gens0, &mut t_prove0);
        let prove_time = t0.elapsed();
        eprintln!("[4/7] SNARK encode={:.2?}, prove pi0={:.2?}", encode_time, prove_time);

        // --- Step 5: verify π⁰ ---
        let t0 = Instant::now();
        let mut t_verify0 = Transcript::new(b"ZKChallenge_C0");
        proof0
            .verify(&comm0, &ai0, &mut t_verify0, &gens0)
            .expect("pi0 verification must succeed");
        eprintln!("[5/7] verify pi0 in {:.2?}", t0.elapsed());

        // --- Step 6: encode + prove + verify π* (C* = C⁰ identity, witness via R1CS solver) ---
        let t0 = Instant::now();
        let (vars_star, public_io_star) = solve_witness_from_r1cs(
            &c0.to_spartan_instance(),
            ch.num_circuit_outputs,
            &x_eval,
        )
        .expect("R1CS solver must succeed for identity circuit");
        let inst_star =
            Instance::new(c0.num_cons, c0.num_vars, c0.num_inputs, &c0.A, &c0.B, &c0.C).unwrap();
        let gens_star =
            SNARKGens::new(c0.num_cons, c0.num_vars, c0.num_inputs, c0.num_non_zero());
        let (comm_star, decomm_star) = SNARK::encode(&inst_star, &gens_star);

        let vs_bytes: Vec<[u8; 32]> = vars_star.iter().map(|s| s.to_bytes()).collect();
        let ios_bytes: Vec<[u8; 32]> = public_io_star.iter().map(|s| s.to_bytes()).collect();
        let av_star = VarsAssignment::new(&vs_bytes).unwrap();
        let ai_star = InputsAssignment::new(&ios_bytes).unwrap();

        let mut t_prove_star = Transcript::new(b"ZKChallenge_Cstar");
        let proof_star = SNARK::prove(
            &inst_star,
            &comm_star,
            &decomm_star,
            av_star,
            &ai_star,
            &gens_star,
            &mut t_prove_star,
        );
        let prove_star_time = t0.elapsed();

        let t0 = Instant::now();
        let mut t_verify_star = Transcript::new(b"ZKChallenge_Cstar");
        proof_star
            .verify(&comm_star, &ai_star, &mut t_verify_star, &gens_star)
            .expect("pi* verification must succeed");
        eprintln!("[6/7] encode+prove pi*={:.2?}, verify pi*={:.2?}", prove_star_time, t0.elapsed());

        // --- Step 7: output equivalence ---
        let y0_pub = &public_io0[..ch.num_circuit_outputs];
        let y_star_pub = &public_io_star[..ch.num_circuit_outputs];
        assert_eq!(y0_pub, y_star_pub, "C0 and C* must produce the same outputs");
        eprintln!("[7/7] output equivalence OK ({} output scalars match)", y0_pub.len());

        eprintln!("=== PASSED in {:.2?} ===\n", total.elapsed());
    }

    // ----- Integration test: full Spartan prove/verify with alias optimization -----

    #[test]
    fn test_alias_optimizer_roundtrip() {
        let total = Instant::now();
        eprintln!("\n=== Alias Optimizer Roundtrip (delta=1) ===");

        // --- Step 1: generate challenge ---
        let t0 = Instant::now();
        let ch = make_challenge(1);
        eprintln!("[1/4] generate challenge: {} constraints in {:.2?}",
            ch.circuit_c0.num_cons, t0.elapsed());

        // --- Step 2: define optimizer ---
        fn alias_optimizer(c0: &Circuit) -> Circuit {
            let si = c0.to_spartan_instance();
            let optimized = remove_aliases(&si);
            Circuit::from_spartan_instance(optimized)
        }

        // --- Step 3: solve (optimize + witnesses + proofs) ---
        let t0 = Instant::now();
        let solution = solve_challenge(&ch, alias_optimizer)
            .expect("solve_challenge must succeed with alias optimizer");
        eprintln!("[2/4] solve_challenge: C* has {} constraints (reduced from {}), in {:.2?}",
            solution.circuit_star.num_cons, ch.circuit_c0.num_cons, t0.elapsed());

        // Sanity: K* < K0
        assert!(
            solution.circuit_star.num_cons < ch.circuit_c0.num_cons,
            "Alias optimizer must reduce constraint count: {} >= {}",
            solution.circuit_star.num_cons, ch.circuit_c0.num_cons
        );

        // --- Step 4: verify ---
        let t0 = Instant::now();
        ch.verify_solution(&solution)
            .expect("verify_solution must succeed for alias-optimized circuit");
        eprintln!("[3/4] verify_solution OK in {:.2?}", t0.elapsed());

        let epsilon = 1.0 - (solution.circuit_star.num_cons as f64 / ch.circuit_c0.num_cons as f64);
        eprintln!("[4/4] epsilon = {:.4} ({} -> {} constraints)",
            epsilon, ch.circuit_c0.num_cons, solution.circuit_star.num_cons);

        eprintln!("=== PASSED in {:.2?} ===\n", total.elapsed());
    }
}
