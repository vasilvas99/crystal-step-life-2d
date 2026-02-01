/// A better implementation of ref_engine.rs
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum CellState {
    Empty,
    Diffusing,
    Solid,
}

struct Grid {
    width: usize,
    height: usize,
    cells: Vec<CellState>,
    solid_indices: Vec<usize>,
    diffusing_indices: Vec<usize>,
}
impl Grid {
    fn new(width: usize, height: usize, c0: f64) -> Self {
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
            .map(|&idx| self.index_to_coords(idx).1)
            .min()
            .unwrap_or(self.height)
    }

    fn collect_empty_positions_above_row(&self, row_limit: usize) -> Vec<usize> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(idx, state)| {
                **state == CellState::Empty && self.index_to_coords(*idx).1 < row_limit
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    fn count_diffusing_above_row(&self, row_limit: usize) -> usize {
        self.diffusing_indices
            .iter()
            .filter(|&&idx| self.index_to_coords(idx).1 < row_limit)
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
            let (x, y) = self.index_to_coords(idx);
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
            let (x, y) = self.index_to_coords(flat_index);
            self.set_cell(x, y, CellState::Empty);
        }

        for &idx in &indices_to_remove {
            self.diffusing_indices.swap_remove(idx);
        }
    }

    fn adjust_diffusing_concentration(&mut self, target_c0: f64, previous_c0: f64) {
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

    fn calculate_random_move_claim(&self, x: usize, y: usize, rng: &mut impl Rng) -> (usize, usize, bool) {
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

    fn calculate_target_positions(&self, shuffled_indices: &[usize]) -> (Vec<(usize, usize)>, Vec<bool>) {
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
                let (x, y) = self.index_to_coords(index);

                let (new_x, new_y, can_move_local) = self.calculate_random_move_claim(x, y, &mut local_rng);
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
            let (x, y) = self.index_to_coords(index);

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

    fn diffuse(&mut self) {
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
}

