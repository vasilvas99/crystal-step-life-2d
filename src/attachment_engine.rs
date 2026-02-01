use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CellState {
    Empty,
    Diffusing,
    Solid,
}

pub struct Grid {
    width: usize,
    height: usize,
    cells: Vec<CellState>,
    solid_indices: Vec<usize>,
    diffusing_indices: Vec<usize>,
}

pub struct GridView<'a> {
    grid: &'a Grid,
}

impl<'a> std::ops::Index<(usize, usize)> for GridView<'a> {
    type Output = CellState;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (x, y) = index;
        self.grid.get_cell(x, y)
    }
}

impl Grid {
    pub fn new(width: usize, height: usize, c0: f64) -> Self {
        let mut grid = Self {
            width,
            height,
            cells: vec![CellState::Empty; width * height],
            solid_indices: Vec::new(),
            diffusing_indices: Vec::new(),
        };

        // set bottom row to Solid
        for x in 0..width {
            grid.set_cell(x, height - 1, CellState::Solid);
            grid.solid_indices.push(grid.get_index(x, height - 1));
        }

        grid.adjust_diffusing_concentration(c0, 0.0);

        grid
    }
    fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn set_cell(&mut self, x: usize, y: usize, state: CellState) {
        let index = self.get_index(x, y);
        self.cells[index] = state;
    }

    fn get_cell(&self, x: usize, y: usize) -> &CellState {
        let index = self.get_index(x, y);
        &self.cells[index]
    }

    fn get_cell_by_index(&self, index: usize) -> CellState {
        self.cells[index]
    }

    fn index_to_coords(&self, index: usize) -> (usize, usize) {
        (index / self.width, index % self.width)
    }

    fn find_topmost_solid_row(&self) -> usize {
        self.solid_indices
            .iter()
            .map(|&idx| self.index_to_coords(idx).0)
            .min()
            .unwrap_or(self.height)
    }

    fn collect_empty_positions_above_row(&self, row_limit: usize) -> Vec<usize> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(idx, state)| {
                **state == CellState::Empty && self.index_to_coords(*idx).0 < row_limit
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    fn count_diffusing_above_row(&self, row_limit: usize) -> usize {
        self.diffusing_indices
            .iter()
            .filter(|&&idx| self.index_to_coords(idx).0 < row_limit)
            .count()
    }

    fn calculate_concentration_delta(&self, target_c0: f64) -> i64 {
        let total_empty = self
            .cells
            .iter()
            .filter(|&&state| state == CellState::Empty)
            .count();
        let total_diffusing = self.diffusing_indices.len();
        let total_available = total_empty + total_diffusing;
        let target_diffusing = (total_available as f64 * target_c0).round() as usize;
        target_diffusing as i64 - total_diffusing as i64
    }

    fn add_diffusing_atoms(&mut self, empty_positions: Vec<usize>, count: usize) {
        if empty_positions.is_empty() {
            return;
        }

        let mut rng = rand::thread_rng();
        let mut positions = empty_positions;
        positions.shuffle(&mut rng);

        // Add as many as possible, up to the requested count
        let actual_count = count.min(positions.len());
        for &idx in positions.iter().take(actual_count) {
            let (y, x) = self.index_to_coords(idx);
            self.set_cell(x, y, CellState::Diffusing);
            self.diffusing_indices.push(idx);
        }
    }

    fn remove_diffusing_atoms(&mut self, count: usize) {
        if count == 0 || count > self.diffusing_indices.len() {
            return;
        }

        let mut rng = rand::thread_rng();
        let mut indices_to_remove: Vec<usize> = (0..self.diffusing_indices.len()).collect();
        indices_to_remove.shuffle(&mut rng);

        indices_to_remove.truncate(count);
        indices_to_remove.sort_unstable_by(|a, b| b.cmp(a));

        for &idx in &indices_to_remove {
            let flat_index = self.diffusing_indices[idx];
            let (y, x) = self.index_to_coords(flat_index);
            self.set_cell(x, y, CellState::Empty);
        }

        for &idx in &indices_to_remove {
            self.diffusing_indices.swap_remove(idx);
        }
    }

    pub fn adjust_diffusing_concentration(&mut self, target_c0: f64, previous_c0: f64) {
        let delta = self.calculate_concentration_delta(target_c0);

        match delta {
            d if d > 0 => {
                // Need to add atoms - only add in empty positions above topmost solid
                let topmost_solid_row = self.find_topmost_solid_row();
                let empty_positions = self.collect_empty_positions_above_row(topmost_solid_row);
                self.add_diffusing_atoms(empty_positions, d as usize);
            }
            d if d < 0 && target_c0 < previous_c0 => {
                // Need to remove atoms - can remove from anywhere
                // Only remove if concentration actually decreased
                self.remove_diffusing_atoms((-d) as usize);
            }
            _ => {}
        }
    }

    fn calculate_random_move_claim(
        &self,
        x: usize,
        y: usize,
        rng: &mut impl Rng,
    ) -> (usize, usize, bool) {
        let step_x = rng.gen_range(-1..=1);
        let step_y = rng.gen_range(-1..=1);

        let new_x = x as i32 + step_x;
        let new_y = y as i32 + step_y;

        // Check if target is valid (within bounds and not solid, diffusing conflicts handled later)
        let can_move = new_x >= 0
            && (new_x as usize) < self.height
            && new_y >= 0
            && (new_y as usize) < self.width
            && self.cells[self.get_index(new_x as usize, new_y as usize)] != CellState::Solid;

        (new_x as usize, new_y as usize, can_move)
    }

    fn calculate_target_positions(
        &self,
        shuffled_indices: &[usize],
    ) -> (Vec<(usize, usize)>, Vec<bool>) {
        let n_atoms = shuffled_indices.len();
        let mut target_positions: Vec<(usize, usize)> = vec![(0, 0); n_atoms];
        let mut can_move: Vec<bool> = vec![false; n_atoms];

        target_positions
            .par_iter_mut()
            .zip(can_move.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (target, can_move_flag))| {
                let mut local_rng = rand::thread_rng();
                let index = shuffled_indices[idx];
                let (y, x) = self.index_to_coords(index);

                let (new_x, new_y, can_move_local) =
                    self.calculate_random_move_claim(x, y, &mut local_rng);
                *target = (new_x, new_y);
                *can_move_flag = can_move_local;
            });

        (target_positions, can_move)
    }

    fn apply_moves(
        &mut self,
        shuffled_indices: &[usize],
        target_positions: &[(usize, usize)],
        can_move: &[bool],
    ) {
        let mut claimed = vec![false; self.width * self.height];
        let mut new_diffusing_list = Vec::with_capacity(shuffled_indices.len());

        for idx in 0..shuffled_indices.len() {
            let index = shuffled_indices[idx];
            let (y, x) = self.index_to_coords(index);

            if can_move[idx] {
                let (new_x, new_y) = target_positions[idx];
                let new_index = self.get_index(new_x, new_y);

                // Check if position is empty and not yet claimed
                if self.cells[new_index] == CellState::Empty && !claimed[new_index] {
                    // Move succeeds
                    self.set_cell(x, y, CellState::Empty);
                    self.set_cell(new_x, new_y, CellState::Diffusing);
                    claimed[new_index] = true;
                    new_diffusing_list.push(new_index);
                } else {
                    // Move blocked - stay in place
                    new_diffusing_list.push(index);
                }
            } else {
                // Can't move (out of bounds or solid) - stay in place
                new_diffusing_list.push(index);
            }
        }

        self.diffusing_indices = new_diffusing_list;
    }

    pub fn diffuse(&mut self) {
        let n_atoms = self.diffusing_indices.len();

        if n_atoms == 0 {
            return;
        }

        // Phase 1: Shuffle indices to ensure fairness (serial)
        let mut shuffled_indices = self.diffusing_indices.clone();
        let mut rng = rand::thread_rng();
        shuffled_indices.shuffle(&mut rng);

        // Phase 2: Calculate target positions (parallel)
        let (target_positions, can_move) = self.calculate_target_positions(&shuffled_indices);

        // Phase 3: Apply moves (serial, with conflict resolution)
        self.apply_moves(&shuffled_indices, &target_positions, &can_move);
    }

    /// Check if a cell has solid neighbors and determine if it's a kink site
    /// Returns (is_kink, has_solid_neighbor)
    fn check_neighbors(&self, x: usize, y: usize) -> (bool, bool) {
        let below = if x + 1 < self.height {
            self.get_cell_by_index(self.get_index(x + 1, y)) == CellState::Solid
        } else {
            false
        };

        let above = if x > 0 {
            self.get_cell_by_index(self.get_index(x - 1, y)) == CellState::Solid
        } else {
            false
        };

        let left = if y > 0 {
            self.get_cell_by_index(self.get_index(x, y - 1)) == CellState::Solid
        } else {
            false
        };

        let right = if y + 1 < self.width {
            self.get_cell_by_index(self.get_index(x, y + 1)) == CellState::Solid
        } else {
            false
        };

        let is_kink = below && (left || right);
        let has_solid_neighbor = above || below || left || right;

        (is_kink, has_solid_neighbor)
    }

    fn calculate_attachment_probability(&self, x: usize, y: usize, pa: f64, pk: f64) -> f64 {
        let (is_kink, has_solid_neighbor) = self.check_neighbors(x, y);

        if is_kink {
            pk
        } else if has_solid_neighbor {
            pa
        } else {
            0.0
        }
    }

    fn calculate_periodic_attachment_probability(
        &self,
        x: usize,
        y: usize,
        pa_max: f64,
        pk: f64,
        pa_k: f64,
    ) -> f64 {
        let (is_kink, has_solid_neighbor) = self.check_neighbors(x, y);

        if is_kink {
            pk
        } else if has_solid_neighbor {
            pa_max
                * (pa_k * 2.0 * std::f64::consts::PI * y as f64 / self.width as f64)
                    .sin()
                    .powi(2)
        } else {
            0.0
        }
    }

    fn calculate_solidification_decisions(&self, pa: f64, pk: f64) -> Vec<bool> {
        let n_atoms = self.diffusing_indices.len();
        let mut should_solidify = vec![false; n_atoms];

        should_solidify
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, should_solidify_flag)| {
                let mut local_rng = rand::thread_rng();
                let index = self.diffusing_indices[idx];
                let (y, x) = self.index_to_coords(index);

                let probability = self.calculate_attachment_probability(x, y, pa, pk);
                if probability > 0.0 && local_rng.gen_bool(probability) {
                    *should_solidify_flag = true;
                }
            });

        should_solidify
    }

    fn calculate_periodic_solidification_decisions(
        &self,
        pa_max: f64,
        pk: f64,
        pa_k: f64,
    ) -> Vec<bool> {
        let n_atoms = self.diffusing_indices.len();
        let mut should_solidify = vec![false; n_atoms];

        should_solidify
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, should_solidify_flag)| {
                let mut local_rng = rand::thread_rng();
                let index = self.diffusing_indices[idx];
                let (y, x) = self.index_to_coords(index);

                let probability =
                    self.calculate_periodic_attachment_probability(x, y, pa_max, pk, pa_k);
                if probability > 0.0 && local_rng.gen_bool(probability) {
                    *should_solidify_flag = true;
                }
            });

        should_solidify
    }

    /// Apply solidification decisions (serial phase)
    fn apply_solidification(&mut self, should_solidify: &[bool]) {
        let mut new_diffusing_list = Vec::with_capacity(self.diffusing_indices.len());

        for idx in 0..self.diffusing_indices.len() {
            let index = self.diffusing_indices[idx];
            if should_solidify[idx] {
                let (y, x) = self.index_to_coords(index);
                self.set_cell(x, y, CellState::Solid);
                self.solid_indices.push(index);
            } else {
                new_diffusing_list.push(index);
            }
        }

        self.diffusing_indices = new_diffusing_list;
    }

    pub fn solidify(&mut self, pa: f64, pk: f64) {
        if self.diffusing_indices.is_empty() {
            return;
        }

        let should_solidify = self.calculate_solidification_decisions(pa, pk);
        self.apply_solidification(&should_solidify);
    }

    /// Perform solidification step with periodic attachment probability potential
    /// pa(y) = pa_max * sin^2(k * 2π * y / width)
    pub fn solidify_periodic(&mut self, pa_max: f64, pk: f64, pa_k: f64) {
        if self.diffusing_indices.is_empty() {
            return;
        }

        let should_solidify = self.calculate_periodic_solidification_decisions(pa_max, pk, pa_k);
        self.apply_solidification(&should_solidify);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_initialization() {
        let grid = Grid::new(100, 100, 0.5);
        assert_eq!(grid.width, 100);
        assert_eq!(grid.height, 100);
        assert_eq!(grid.cells.len(), 10000);

        // Check that bottom row is solid
        for x in 0..100 {
            assert_eq!(grid.get_cell(x, 99), &CellState::Solid);
        }

        // Check that solid_indices contains bottom row
        assert_eq!(grid.solid_indices.len(), 100);
    }

    #[test]
    fn test_grid_concentration_initialization() {
        let grid = Grid::new(100, 100, 0.3);

        let empty_count = grid.cells.iter().filter(|&&s| s == CellState::Empty).count();
        let diffusing_count = grid.diffusing_indices.len();
        let solid_count = grid.solid_indices.len();

        // Total should equal grid size
        assert_eq!(empty_count + diffusing_count + solid_count, 10000);

        // Concentration should be approximately 0.3
        let total_available = empty_count + diffusing_count;
        let actual_c0 = diffusing_count as f64 / total_available as f64;
        assert!((actual_c0 - 0.3).abs() < 0.01, "Concentration mismatch: expected ~0.3, got {}", actual_c0);
    }

    #[test]
    fn test_index_to_coords_conversion() {
        let grid = Grid::new(100, 100, 0.0);

        // Test various positions
        let test_cases = vec![
            (0, 0, 0),
            (5, 0, 5),
            (0, 1, 100),
            (50, 50, 5050),
            (99, 99, 9999),
        ];

        for (x, y, expected_index) in test_cases {
            let index = grid.get_index(x, y);
            assert_eq!(index, expected_index, "get_index({}, {}) failed", x, y);

            let (restored_y, restored_x) = grid.index_to_coords(index);
            assert_eq!(restored_x, x, "index_to_coords x failed for index {}", index);
            assert_eq!(restored_y, y, "index_to_coords y failed for index {}", index);
        }
    }

    #[test]
    fn test_diffuse_step_preserves_atom_count() {
        let mut grid = Grid::new(50, 50, 0.3);

        let initial_diffusing = grid.diffusing_indices.len();
        let initial_solid = grid.solid_indices.len();
        let initial_empty = grid.cells.iter().filter(|&&s| s == CellState::Empty).count();

        // Perform diffusion step
        grid.diffuse();

        let final_diffusing = grid.diffusing_indices.len();
        let final_solid = grid.solid_indices.len();
        let final_empty = grid.cells.iter().filter(|&&s| s == CellState::Empty).count();

        // Atom counts should be preserved
        assert_eq!(final_diffusing, initial_diffusing, "Diffusing count changed");
        assert_eq!(final_solid, initial_solid, "Solid count changed");
        assert_eq!(final_empty, initial_empty, "Empty count changed");

        // Verify diffusing_indices matches actual grid state
        let actual_diffusing_on_grid = grid.cells.iter()
            .filter(|&&s| s == CellState::Diffusing)
            .count();
        assert_eq!(actual_diffusing_on_grid, final_diffusing, "Diffusing indices out of sync with grid");
    }

    #[test]
    fn test_diffuse_step_no_atoms_stuck_in_solid() {
        let mut grid = Grid::new(50, 50, 0.3);

        // Perform multiple diffusion steps
        for _ in 0..10 {
            grid.diffuse();

            // Check that no diffusing atoms are in solid positions
            for &idx in &grid.diffusing_indices {
                assert_eq!(grid.get_cell_by_index(idx), CellState::Diffusing,
                    "Diffusing atom index points to non-diffusing cell");
            }
        }
    }

    #[test]
    fn test_solidify_step_reduces_or_maintains_diffusing() {
        let mut grid = Grid::new(50, 50, 0.0);

        // Manually add diffusing atoms right above solid row
        for x in 10..20 {
            let y = grid.height - 2;
            grid.set_cell(x, y, CellState::Diffusing);
            grid.diffusing_indices.push(grid.get_index(x, y));
        }

        let initial_diffusing = grid.diffusing_indices.len();
        let initial_solid = grid.solid_indices.len();

        // Run solidification with high probability
        grid.solidify(1.0, 1.0);

        let final_diffusing = grid.diffusing_indices.len();
        let final_solid = grid.solid_indices.len();

        // Diffusing should decrease or stay same
        assert!(final_diffusing <= initial_diffusing, "Diffusing increased");

        // Total atoms should be preserved
        assert_eq!(
            final_solid + final_diffusing,
            initial_solid + initial_diffusing,
            "Total atoms not preserved"
        );

        // If diffusing decreased, solid should increase by same amount
        let diffusing_lost = initial_diffusing - final_diffusing;
        let solid_gained = final_solid - initial_solid;
        assert_eq!(diffusing_lost, solid_gained, "Atom transfer mismatch");
    }

    #[test]
    fn test_solidify_step_never_solidifies_without_neighbors() {
        let mut grid = Grid::new(50, 50, 0.0);

        // Add isolated diffusing atom (not near any solid)
        grid.set_cell(25, 25, CellState::Diffusing);
        grid.diffusing_indices.push(grid.get_index(25, 25));

        let initial_diffusing = grid.diffusing_indices.len();

        // Run solidification many times with high probability
        for _ in 0..100 {
            grid.solidify(1.0, 1.0);
        }

        // Isolated atom should never solidify
        assert_eq!(grid.diffusing_indices.len(), initial_diffusing,
            "Isolated atom solidified without neighbors");
    }

    #[test]
    fn test_solidify_periodic_preserves_atoms() {
        let mut grid = Grid::new(50, 50, 0.0);

        // Add diffusing atoms above solid row
        for x in 10..20 {
            let y = grid.height - 2;
            grid.set_cell(x, y, CellState::Diffusing);
            grid.diffusing_indices.push(grid.get_index(x, y));
        }

        let initial_total = grid.solid_indices.len() + grid.diffusing_indices.len();

        // Run periodic solidification
        grid.solidify_periodic(0.8, 1.0, 2.0);

        // Total atoms should be preserved
        let final_total = grid.solid_indices.len() + grid.diffusing_indices.len();
        assert_eq!(final_total, initial_total, "Total atoms not preserved in periodic solidification");
    }

    #[test]
    fn test_check_neighbors_detects_kink_sites() {
        let mut grid = Grid::new(50, 50, 0.0);

        // Create a kink site pattern:
        // The bottom row (y=49) is already solid
        // Add solids to create a kink above it:
        // .X.  <- test position (test_x-1, test_y) should detect kink
        // XX.  <- (test_x, test_y) and (test_x-1, test_y)
        // SSS  <- bottom row at y=49 (already solid)
        let test_x = 25;
        let test_y = grid.height - 1; // bottom row

        // Add a solid one row above bottom
        grid.set_cell(test_x, test_y - 1, CellState::Solid);

        // Position at (test_x-1, test_y-1) should be a kink site
        // - below: (test_x, test_y-1) = solid ✓
        // - right: (test_x-1, test_y) = solid (bottom row) ✓
        // - kink = below && (left || right) = true
        let (is_kink, has_neighbor) = grid.check_neighbors(test_x - 1, test_y - 1);

        assert!(is_kink, "Failed to detect kink site");
        assert!(has_neighbor, "Failed to detect solid neighbor");
    }

    #[test]
    fn test_check_neighbors_detects_regular_neighbors() {
        let mut grid = Grid::new(50, 50, 0.0);

        // Create a non-kink neighbor:
        // .X.
        // .S.
        // SSS (solid row at bottom)
        let test_x = 25;
        let test_y = grid.height - 2;

        grid.set_cell(test_x, test_y, CellState::Solid);

        // Position above should have neighbor but not be a kink
        let (is_kink, has_neighbor) = grid.check_neighbors(test_x, test_y - 1);

        assert!(!is_kink, "False positive kink detection");
        assert!(has_neighbor, "Failed to detect solid neighbor");
    }

    #[test]
    fn test_adjust_concentration_adds_atoms() {
        let mut grid = Grid::new(50, 50, 0.1);

        let initial_diffusing = grid.diffusing_indices.len();

        // Increase target concentration
        grid.adjust_diffusing_concentration(0.3, 0.1);

        let final_diffusing = grid.diffusing_indices.len();

        // Should have added atoms
        assert!(final_diffusing > initial_diffusing, "Failed to add atoms when increasing concentration");
    }

    #[test]
    fn test_adjust_concentration_removes_atoms() {
        let mut grid = Grid::new(50, 50, 0.5);

        let initial_diffusing = grid.diffusing_indices.len();

        // Decrease target concentration
        grid.adjust_diffusing_concentration(0.2, 0.5);

        let final_diffusing = grid.diffusing_indices.len();

        // Should have removed atoms
        assert!(final_diffusing < initial_diffusing, "Failed to remove atoms when decreasing concentration");
    }

    #[test]
    fn test_find_topmost_solid_row() {
        let mut grid = Grid::new(50, 50, 0.0);

        // Initially only bottom row is solid (row 49)
        let topmost = grid.find_topmost_solid_row();
        assert_eq!(topmost, 49, "Initial topmost should be bottom row");

        // Add solid at row 30
        grid.set_cell(25, 30, CellState::Solid);
        grid.solid_indices.push(grid.get_index(25, 30));

        let topmost = grid.find_topmost_solid_row();
        assert_eq!(topmost, 30, "Failed to find topmost solid at row 30");

        // Add solid at row 20
        grid.set_cell(25, 20, CellState::Solid);
        grid.solid_indices.push(grid.get_index(25, 20));

        let topmost = grid.find_topmost_solid_row();
        assert_eq!(topmost, 20, "Failed to update topmost solid to row 20");
    }

    #[test]
    fn test_no_duplicate_indices() {
        let mut grid = Grid::new(50, 50, 0.3);

        // Perform several steps
        for _ in 0..10 {
            grid.diffuse();
            grid.solidify(0.1, 0.3);
        }

        // Check for duplicate indices in diffusing_indices
        let mut sorted_diffusing = grid.diffusing_indices.clone();
        sorted_diffusing.sort_unstable();
        let unique_count_before = sorted_diffusing.len();
        sorted_diffusing.dedup();
        assert_eq!(sorted_diffusing.len(), unique_count_before,
            "Duplicate indices in diffusing_indices");

        // Check for duplicate indices in solid_indices
        let mut sorted_solid = grid.solid_indices.clone();
        sorted_solid.sort_unstable();
        let unique_solid_before = sorted_solid.len();
        sorted_solid.dedup();
        assert_eq!(sorted_solid.len(), unique_solid_before,
            "Duplicate indices in solid_indices");

        // Check no overlap between diffusing and solid
        for &diff_idx in &grid.diffusing_indices {
            assert!(!grid.solid_indices.contains(&diff_idx),
                "Index {} is in both diffusing and solid lists", diff_idx);
        }
    }

    #[test]
    fn test_grid_consistency_after_multiple_steps() {
        let mut grid = Grid::new(50, 50, 0.4);

        // Run simulation for multiple steps
        for _ in 0..20 {
            grid.diffuse();
            grid.solidify(0.1, 0.3);

            // Verify consistency: indices match grid state
            let mut diffusing_count_from_grid = 0;
            let mut solid_count_from_grid = 0;

            for idx in 0..grid.cells.len() {
                match grid.cells[idx] {
                    CellState::Diffusing => diffusing_count_from_grid += 1,
                    CellState::Solid => solid_count_from_grid += 1,
                    CellState::Empty => {}
                }
            }

            assert_eq!(diffusing_count_from_grid, grid.diffusing_indices.len(),
                "Diffusing count mismatch between grid and indices");
            assert_eq!(solid_count_from_grid, grid.solid_indices.len(),
                "Solid count mismatch between grid and indices");

            // Verify all indices point to correct cell states
            for &idx in &grid.diffusing_indices {
                assert_eq!(grid.cells[idx], CellState::Diffusing,
                    "Diffusing index {} points to {:?}", idx, grid.cells[idx]);
            }

            for &idx in &grid.solid_indices {
                assert_eq!(grid.cells[idx], CellState::Solid,
                    "Solid index {} points to {:?}", idx, grid.cells[idx]);
            }
        }
    }

    #[test]
    fn test_solidify_with_zero_probability_never_solidifies() {
        let mut grid = Grid::new(50, 50, 0.0);

        // Add diffusing atoms above solid
        for x in 10..20 {
            let y = grid.height - 2;
            grid.set_cell(x, y, CellState::Diffusing);
            grid.diffusing_indices.push(grid.get_index(x, y));
        }

        let initial_diffusing = grid.diffusing_indices.len();

        // Run solidification with zero probability many times
        for _ in 0..100 {
            grid.solidify(0.0, 0.0);
        }

        // No atoms should solidify
        assert_eq!(grid.diffusing_indices.len(), initial_diffusing,
            "Atoms solidified with zero probability");
    }

    #[test]
    fn test_empty_grid_operations() {
        let mut grid = Grid::new(50, 50, 0.0);

        // Clear all diffusing atoms
        grid.diffusing_indices.clear();

        // These should not panic
        grid.diffuse();
        grid.solidify(0.5, 0.8);
        grid.solidify_periodic(0.5, 0.8, 2.0);

        // Grid should remain consistent
        assert_eq!(grid.diffusing_indices.len(), 0);
    }

    #[test]
    fn test_coordinate_system_consistency() {
        let grid = Grid::new(100, 100, 0.0);

        // The bottom row should be y=99, not y=0
        for x in 0..100 {
            let idx = grid.get_index(x, 99);
            let (restored_y, restored_x) = grid.index_to_coords(idx);
            assert_eq!(restored_x, x);
            assert_eq!(restored_y, 99);
            assert_eq!(grid.cells[idx], CellState::Solid, "Bottom row should be solid");
        }
    }

    #[test]
    fn test_find_topmost_uses_correct_coordinate() {
        let mut grid = Grid::new(50, 50, 0.0);

        // Bottom row is y=49 (highest y value)
        // Add a solid at y=10 (lower y, but "topmost" in visual sense)
        grid.set_cell(25, 10, CellState::Solid);
        grid.solid_indices.push(grid.get_index(25, 10));

        let topmost = grid.find_topmost_solid_row();

        // Should return 10 (minimum y), not 49
        assert_eq!(topmost, 10, "find_topmost_solid_row should return minimum y coordinate");
    }
}
