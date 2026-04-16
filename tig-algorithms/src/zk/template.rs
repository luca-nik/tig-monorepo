/*!
Copyright [year copyright work created] [name of copyright owner]

Identity of Submitter [name of person or entity that submits the Work to TIG]

UAI [UAI (if applicable)]

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use tig_challenges::zk::*;

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    // Optimize C0 -> C* with fewer constraints, then let solve_challenge handle
    // witness computation and proof generation automatically.
    //
    // Your optimizer must return a SpartanInstance with:
    //   1. Strictly fewer constraints: C*.num_cons < C0.num_cons
    //   2. Same function: same outputs for the evaluation point x_eval
    //   3. Rows in topological evaluation order (each row has at most one
    //      unknown when processed in sequence — see OptimizeCircuitFn docs)
    //
    // Use baselines::remove_aliases as a starting point (~13% reduction).
    // Aim for >50% by also handling Scale nodes, Pow5 sharing, and CSE.

    let solution = tig_challenges::zk::solve_challenge(challenge, optimize)?;
    Ok(Some(solution))
}

fn optimize(c0: &SpartanInstance) -> SpartanInstance {
    // Replace this with your optimizer.
    // remove_aliases handles Alias constraints only (~13% reduction).
    baselines::remove_aliases(c0)
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
