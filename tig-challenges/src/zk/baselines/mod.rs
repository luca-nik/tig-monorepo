use crate::zk::r1cs::{R1CSMatrix, SpartanInstance};
use curve25519_dalek::scalar::Scalar;
use std::collections::HashMap;

/// Removes alias constraints from an R1CS instance via variable substitution.
///
/// An alias constraint has the form `out * 1 = src`:
///   - A: single entry `(col_out, 1)`, col_out < num_vars (private)
///   - B: single entry `(const_col, 1)`
///   - C: single entry `(col_src, 1)`
///
/// The function substitutes `col_out → col_src` everywhere, removes the alias
/// rows, and compacts private variable columns so `num_vars` shrinks.
///
/// Preserves topological row order. Pure function — returns a clone if no
/// aliases are found.
///
/// ~13% constraint reduction on default-configuration circuits.
pub fn remove_aliases(instance: &SpartanInstance) -> SpartanInstance {
    let num_vars = instance.num_vars;
    let const_col = num_vars;
    let one_bytes = Scalar::ONE.to_bytes();

    // --- Build per-row views ---
    let mut a_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];
    let mut b_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];
    let mut c_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];

    for &(row, col, bytes) in &instance.A { a_rows[row].push((col, bytes)); }
    for &(row, col, bytes) in &instance.B { b_rows[row].push((col, bytes)); }
    for &(row, col, bytes) in &instance.C { c_rows[row].push((col, bytes)); }

    // --- Detect alias rows and build substitution map ---
    let mut substitution: HashMap<usize, usize> = HashMap::new();
    let mut removed_rows: Vec<bool> = vec![false; instance.num_cons];

    for row in 0..instance.num_cons {
        if a_rows[row].len() != 1 || b_rows[row].len() != 1 || c_rows[row].len() != 1 {
            continue;
        }
        let (a_col, a_val) = a_rows[row][0];
        let (b_col, b_val) = b_rows[row][0];
        let (c_col, c_val) = c_rows[row][0];

        if a_val != one_bytes || b_val != one_bytes || c_val != one_bytes { continue; }
        if b_col != const_col { continue; }

        // a_col = c_col; substitute away whichever is private
        if a_col < num_vars {
            substitution.insert(a_col, c_col);
            removed_rows[row] = true;
        } else if c_col < num_vars {
            substitution.insert(c_col, a_col);
            removed_rows[row] = true;
        }
    }

    if substitution.is_empty() {
        return instance.clone();
    }

    // --- Resolve substitution chains: a → b → c flattened to a → c ---
    let mut changed = true;
    while changed {
        changed = false;
        let snap: Vec<(usize, usize)> = substitution.iter().map(|(&k, &v)| (k, v)).collect();
        for (key, target) in snap {
            if let Some(&further) = substitution.get(&target) {
                substitution.insert(key, further);
                changed = true;
            }
        }
    }

    let remap_col = |col: usize| substitution.get(&col).copied().unwrap_or(col);

    // --- Apply substitutions to surviving rows ---
    let mut new_a: R1CSMatrix = Vec::new();
    let mut new_b: R1CSMatrix = Vec::new();
    let mut new_c: R1CSMatrix = Vec::new();
    let mut new_row = 0usize;

    for row in 0..instance.num_cons {
        if removed_rows[row] { continue; }

        emit_row(&a_rows[row], &mut new_a, new_row, &remap_col);
        emit_row(&b_rows[row], &mut new_b, new_row, &remap_col);
        emit_row(&c_rows[row], &mut new_c, new_row, &remap_col);
        new_row += 1;
    }

    let new_num_cons = new_row;

    // --- Column compaction (private variables only) ---
    let live_private: Vec<usize> = {
        use std::collections::HashSet;
        let mut set: HashSet<usize> = HashSet::new();
        for &(_, col, _) in new_a.iter().chain(new_b.iter()).chain(new_c.iter()) {
            if col < num_vars { set.insert(col); }
        }
        let mut v: Vec<usize> = set.into_iter().collect();
        v.sort();
        v
    };
    let new_num_vars = live_private.len();

    let mut col_remap: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_col) in live_private.iter().enumerate() {
        col_remap.insert(old_col, new_idx);
    }
    col_remap.insert(const_col, new_num_vars);
    for i in 0..instance.num_inputs {
        col_remap.insert(num_vars + 1 + i, new_num_vars + 1 + i);
    }

    for entry in new_a.iter_mut().chain(new_b.iter_mut()).chain(new_c.iter_mut()) {
        entry.1 = col_remap[&entry.1];
    }

    SpartanInstance {
        num_cons: new_num_cons,
        num_vars: new_num_vars,
        num_inputs: instance.num_inputs,
        A: new_a,
        B: new_b,
        C: new_c,
    }
}

fn emit_row(
    src: &[(usize, [u8; 32])],
    dest: &mut R1CSMatrix,
    new_row: usize,
    remap: &dyn Fn(usize) -> usize,
) {
    let mut merged: HashMap<usize, Scalar> = HashMap::new();
    for &(col, bytes) in src {
        let new_col = remap(col);
        let val = Scalar::from_canonical_bytes(bytes).unwrap();
        *merged.entry(new_col).or_insert(Scalar::ZERO) += val;
    }
    for (col, val) in merged {
        if val != Scalar::ZERO {
            dest.push((new_row, col, val.to_bytes()));
        }
    }
}

// =============================================================================
// remove_scales
// =============================================================================

/// Removes scale constraints from an R1CS instance via variable substitution.
///
/// A scale constraint has the form `(k * src) * 1 = out`:
///   - A: single entry `(col_src, k)`, k ≠ ONE
///   - B: single entry `(const_col, ONE)`
///   - C: single entry `(col_out, ONE)`, col_out < num_vars (private)
///
/// The function substitutes `col_out → k * col_src` everywhere — any entry
/// `(row, col_out, v)` becomes `(row, col_src, v * k)` — removes the scale
/// rows, and compacts private variable columns so `num_vars` shrinks.
///
/// Preserves topological row order. Pure function — returns a clone if no
/// scale constraints are found.
///
/// ~20% constraint reduction on default-configuration circuits (on top of
/// alias removal).
pub fn remove_scales(instance: &SpartanInstance) -> SpartanInstance {
    let num_vars = instance.num_vars;
    let const_col = num_vars;
    let one_bytes = Scalar::ONE.to_bytes();

    // --- Build per-row views ---
    let mut a_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];
    let mut b_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];
    let mut c_rows: Vec<Vec<(usize, [u8; 32])>> = vec![Vec::new(); instance.num_cons];

    for &(row, col, bytes) in &instance.A { a_rows[row].push((col, bytes)); }
    for &(row, col, bytes) in &instance.B { b_rows[row].push((col, bytes)); }
    for &(row, col, bytes) in &instance.C { c_rows[row].push((col, bytes)); }

    // --- Detect scale rows and build substitution map ---
    // substitution: col_out → (col_src, k)  meaning  out = k * src
    let mut substitution: HashMap<usize, (usize, Scalar)> = HashMap::new();
    let mut removed_rows: Vec<bool> = vec![false; instance.num_cons];

    for row in 0..instance.num_cons {
        if a_rows[row].len() != 1 || b_rows[row].len() != 1 || c_rows[row].len() != 1 {
            continue;
        }
        let (a_col, a_val) = a_rows[row][0];
        let (b_col, b_val) = b_rows[row][0];
        let (c_col, c_val) = c_rows[row][0];

        // B must be the constant wire with coefficient 1
        if b_col != const_col || b_val != one_bytes { continue; }
        // C must be a private variable with coefficient 1 (this is the output)
        if c_col >= num_vars || c_val != one_bytes { continue; }
        // A coefficient must not be ONE (alias rows already handled by remove_aliases)
        if a_val == one_bytes { continue; }
        // Avoid degenerate self-loops
        if a_col == c_col { continue; }

        let k = Scalar::from_canonical_bytes(a_val).unwrap();
        if k == Scalar::ZERO { continue; }

        substitution.insert(c_col, (a_col, k));
        removed_rows[row] = true;
    }

    if substitution.is_empty() {
        return instance.clone();
    }

    // --- Resolve substitution chains: out→(mid,k1), mid→(src,k2) ⟹ out→(src,k1*k2) ---
    let mut changed = true;
    while changed {
        changed = false;
        let snap: Vec<(usize, (usize, Scalar))> =
            substitution.iter().map(|(&k, &v)| (k, v)).collect();
        for (key, (target_col, k1)) in snap {
            if let Some(&(final_col, k2)) = substitution.get(&target_col) {
                substitution.insert(key, (final_col, k1 * k2));
                changed = true;
            }
        }
    }

    // --- Apply substitutions to surviving rows ---
    // Entry (col, coeff) becomes (new_col, coeff * k) when col is in the substitution map.
    let remap = |col: usize, coeff: Scalar| -> (usize, Scalar) {
        if let Some(&(new_col, k)) = substitution.get(&col) {
            (new_col, coeff * k)
        } else {
            (col, coeff)
        }
    };

    let mut new_a: R1CSMatrix = Vec::new();
    let mut new_b: R1CSMatrix = Vec::new();
    let mut new_c: R1CSMatrix = Vec::new();
    let mut new_row = 0usize;

    for row in 0..instance.num_cons {
        if removed_rows[row] { continue; }
        emit_row_scaled(&a_rows[row], &mut new_a, new_row, &remap);
        emit_row_scaled(&b_rows[row], &mut new_b, new_row, &remap);
        emit_row_scaled(&c_rows[row], &mut new_c, new_row, &remap);
        new_row += 1;
    }

    let new_num_cons = new_row;

    // --- Column compaction (private variables only) ---
    let live_private: Vec<usize> = {
        use std::collections::HashSet;
        let mut set: HashSet<usize> = HashSet::new();
        for &(_, col, _) in new_a.iter().chain(new_b.iter()).chain(new_c.iter()) {
            if col < num_vars { set.insert(col); }
        }
        let mut v: Vec<usize> = set.into_iter().collect();
        v.sort();
        v
    };
    let new_num_vars = live_private.len();

    let mut col_remap: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_col) in live_private.iter().enumerate() {
        col_remap.insert(old_col, new_idx);
    }
    col_remap.insert(const_col, new_num_vars);
    for i in 0..instance.num_inputs {
        col_remap.insert(num_vars + 1 + i, new_num_vars + 1 + i);
    }

    for entry in new_a.iter_mut().chain(new_b.iter_mut()).chain(new_c.iter_mut()) {
        entry.1 = col_remap[&entry.1];
    }

    SpartanInstance {
        num_cons: new_num_cons,
        num_vars: new_num_vars,
        num_inputs: instance.num_inputs,
        A: new_a,
        B: new_b,
        C: new_c,
    }
}

/// Removes both alias and scale constraints: runs [`remove_aliases`] then
/// [`remove_scales`] on the result.
///
/// Alias removal must run first because scale detection skips rows with
/// coefficient ONE in A — once aliases are gone, the remaining ONE-coefficient
/// rows are genuine Mul/Add operations, not scales.
///
/// ~33% constraint reduction on default-configuration circuits (13% aliases
/// + ~20% scales).
pub fn remove_aliases_and_scales(instance: &SpartanInstance) -> SpartanInstance {
    remove_scales(&remove_aliases(instance))
}

fn emit_row_scaled(
    src: &[(usize, [u8; 32])],
    dest: &mut R1CSMatrix,
    new_row: usize,
    remap: &dyn Fn(usize, Scalar) -> (usize, Scalar),
) {
    let mut merged: HashMap<usize, Scalar> = HashMap::new();
    for &(col, bytes) in src {
        let coeff = Scalar::from_canonical_bytes(bytes).unwrap();
        let (new_col, new_coeff) = remap(col, coeff);
        *merged.entry(new_col).or_insert(Scalar::ZERO) += new_coeff;
    }
    for (col, val) in merged {
        if val != Scalar::ZERO {
            dest.push((new_row, col, val.to_bytes()));
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::{Challenge, Difficulty, crypto::CryptoHash, r1cs::solve_witness_forward};

    fn make_challenge() -> Challenge {
        let mut seed = [0u8; 32];
        seed[0] = 42;
        Challenge::generate_instance(&seed, &Difficulty { delta: 1 }).unwrap()
    }

    #[test]
    fn test_remove_scales_reduces_constraints() {
        let ch = make_challenge();
        let c0 = &ch.circuit_c0;

        let after_aliases = remove_aliases(c0);
        let after_both = remove_aliases_and_scales(c0);

        assert!(
            after_both.num_cons < c0.num_cons,
            "remove_aliases_and_scales must reduce constraints: {} -> {}",
            c0.num_cons, after_both.num_cons
        );
        assert!(
            after_both.num_cons <= after_aliases.num_cons,
            "remove_aliases_and_scales must be at least as good as remove_aliases: {} vs {}",
            after_both.num_cons, after_aliases.num_cons
        );

        let eps_aliases = 1.0 - after_aliases.num_cons as f64 / c0.num_cons as f64;
        let eps_both = 1.0 - after_both.num_cons as f64 / c0.num_cons as f64;
        eprintln!(
            "[remove_scales] C0: {} → aliases: {} (ε={:.3}) → aliases+scales: {} (ε={:.3})",
            c0.num_cons, after_aliases.num_cons, eps_aliases, after_both.num_cons, eps_both
        );
    }

    #[test]
    fn test_remove_scales_topological_order() {
        let ch = make_challenge();
        let result = remove_aliases_and_scales(&ch.circuit_c0);

        let h0 = CryptoHash::from_serializable(&ch.circuit_c0).unwrap();
        let x_eval = h0.combine(&h0).to_scalars(ch.num_circuit_inputs);

        let witness = solve_witness_forward(&result, ch.num_circuit_outputs, &x_eval);
        assert!(
            witness.is_ok(),
            "forward witness solver must succeed on aliases+scales result: {:?}",
            witness.err()
        );
        eprintln!("[remove_scales] topological order preserved, witness solved OK");
    }
}
