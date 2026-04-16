/*!
Copyright 2025 luca-nik

Identity of Submitter luca-nik

Licensed under the TIG Inbound Game License v2.0 or (at your option) any later
version (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
*/

use anyhow::Result;
use tig_challenges::zk::*;

pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let solution = tig_challenges::zk::solve_challenge(challenge, optimize)?;
    Ok(Some(solution))
}

fn optimize(c0: &SpartanInstance) -> SpartanInstance {
    baselines::remove_aliases(c0)
}
