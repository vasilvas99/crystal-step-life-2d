use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CellState {
    Empty,
    Diffusing,
    Solid,
}

pub enum AttachmentType {
    ToKink,
    ToAny,
    NoAttachment,
}

#[derive(Clone)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    cells: Vec<CellState>,
    pub solid_indices: Vec<usize>,
    pub diffusing_indices: Vec<usize>,
    parity: bool,
}

pub struct GridView<'a> {
    grid: &'a Grid,
}

impl<'a> GridView<'a> {
    pub fn width(&self) -> usize {
        self.grid.width
    }

    pub fn height(&self) -> usize {
        self.grid.height
    }

    pub fn diffusing_count(&self) -> usize {
        self.grid.diffusing_indices.len()
    }

    pub fn solid_count(&self) -> usize {
        self.grid.solid_indices.len()
    }

    pub fn total_atom_count(&self) -> usize {
        self.diffusing_count() + self.solid_count()
    }

    pub fn get_cell(&self, x: usize, y: usize) -> CellState {
        *self.grid.get_cell(x, y)
    }
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
            parity: false,
        };

        // set bottom row to Solid (x=height-1, all y columns)
        for y in 0..width {
            grid.set_cell(height - 1, y, CellState::Solid);
            grid.solid_indices.push(grid.get_index(height - 1, y));
        }

        grid.adjust_diffusing_concentration(c0, 0.0);

        grid
    }
    pub fn get_index(&self, x: usize, y: usize) -> usize {
        x * self.width + y
    }

    pub fn index_to_coords(&self, index: usize) -> (usize, usize) {
        (index / self.width, index % self.width)
    }

    pub fn set_cell(&mut self, x: usize, y: usize, state: CellState) {
        let index = self.get_index(x, y);
        self.cells[index] = state;
    }

    pub fn get_cell(&self, x: usize, y: usize) -> &CellState {
        let index = self.get_index(x, y);
        &self.cells[index]
    }
    pub fn get_view(&self) -> GridView<'_> {
        GridView { grid: self }
    }
    fn get_cell_by_index(&self, index: usize) -> CellState {
        self.cells[index]
    }

    fn find_topmost_solid_row(&self) -> usize {
        self.solid_indices
            .par_iter()
            .map(|&idx| self.index_to_coords(idx).0)
            .min()
            .unwrap_or(self.height)
    }

    fn collect_empty_positions_above_row(&self, row_limit: usize) -> Vec<usize> {
        self.cells
            .par_iter()
            .enumerate()
            .filter(|(idx, state)| {
                **state == CellState::Empty && self.index_to_coords(*idx).0 < row_limit
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    fn calculate_concentration_delta(&self, target_c0: f64) -> i64 {
        let total_empty = self
            .cells
            .par_iter()
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

        let mut rng = rand::rng();
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

        let mut rng = rand::rng();
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

    /// Diffuse particles using Margolus neighbourhood partitioning.
    ///
    /// The grid is divided into non-overlapping 2x2 blocks. Within each block,
    /// the non-solid cells are randomly shuffled (permuted). Because blocks are
    /// independent, all blocks are processed fully in parallel with no conflict
    /// resolution needed.
    ///
    /// The `parity` flag alternates the block offset between (0,0) and (1,1)
    /// each call, ensuring particles can cross block boundaries over successive
    /// sub-steps.
    ///
    /// **Diffusion rate:** The uniform random permutation within each 2×2 block
    /// yields a higher effective diffusion coefficient than a nearest-neighbour
    /// random walk. Adjust `nds` if a specific diffusion rate is needed.
    /// Boundary rows/columns participate in blocks on only one parity, so edge
    /// particles diffuse at half the interior rate (inherent to Margolus).
    ///
    /// **Important:** This method modifies `cells` in-place but does NOT update
    /// `diffusing_indices`. Callers must either call `rebuild_diffusing_indices`
    /// or rely on `solidify`/`solidify_periodic` (which rebuild it internally).
    pub fn diffuse_margolus(&mut self) {
        let w = self.width;
        let h = self.height;
        let offset = if self.parity { 1 } else { 0 };
        self.parity = !self.parity;

        let row_start = offset;
        let usable_rows = ((h - row_start) / 2) * 2;
        if usable_rows < 2 || w < 2 {
            return;
        }

        let col_start = offset;
        let start = row_start * w;
        let end = (row_start + usable_rows) * w;

        // Boundary rows/columns only participate in blocks on one parity,
        // so edge particles diffuse at half rate — inherent to Margolus.

        self.cells[start..end]
            .par_chunks_mut(2 * w)
            .for_each(|chunk| {
                let mut rng = rand::rng();
                let mut col = col_start;
                while col + 1 < w {
                    let indices = [col, col + 1, w + col, w + col + 1];

                    // Collect non-solid positions in the block
                    let mut movable = [0usize; 4];
                    let mut n = 0;
                    for &i in &indices {
                        if chunk[i] != CellState::Solid {
                            movable[n] = i;
                            n += 1;
                        }
                    }

                    // Fisher-Yates shuffle of non-solid cell states.
                    // Solid cells stay pinned; the remaining states are
                    // randomly permuted, which conserves particle count.
                    if n > 1 {
                        for i in (1..n).rev() {
                            let j = rng.random_range(0..=i);
                            chunk.swap(movable[i], movable[j]);
                        }
                    }

                    col += 2;
                }
            });
    }

    pub fn check_neighbors(&self, x: usize, y: usize) -> AttachmentType {
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

        if is_kink {
            AttachmentType::ToKink
        } else if has_solid_neighbor {
            AttachmentType::ToAny
        } else {
            AttachmentType::NoAttachment
        }
    }

    fn calculate_attachment_probability(&self, x: usize, y: usize, pa: f64, pk: f64) -> f64 {
        let attachment_type = self.check_neighbors(x, y);

        match attachment_type {
            AttachmentType::ToKink => pk,
            AttachmentType::ToAny => pa,
            AttachmentType::NoAttachment => 0.0,
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
        match self.check_neighbors(x, y) {
            AttachmentType::ToKink => pk,
            AttachmentType::ToAny => {
                pa_max
                    * (pa_k * 2.0 * std::f64::consts::PI * y as f64 / self.width as f64)
                        .sin()
                        .powi(2)
            }
            AttachmentType::NoAttachment => 0.0,
        }
    }

    /// Rebuild `diffusing_indices` from the cells array.
    ///
    /// Useful after Margolus diffusion (which shuffles cells in-place)
    /// when the index list needs to be resynchronised with the grid.
    /// Not required before `solidify`/`solidify_periodic`, which scan
    /// cells directly and rebuild the list as a side effect.
    pub fn rebuild_diffusing_indices(&mut self) {
        self.diffusing_indices = self
            .cells
            .par_iter()
            .enumerate()
            .filter(|&(_, &s)| s == CellState::Diffusing)
            .map(|(i, _)| i)
            .collect();
    }

    /// Solidify diffusing particles adjacent to solid cells.
    ///
    /// Scans the `cells` array directly (no dependency on `diffusing_indices`),
    /// then rebuilds `diffusing_indices` as a side effect of the serial apply
    /// phase. This avoids the need for a separate `rebuild_diffusing_indices`
    /// call between diffusion and solidification.
    pub fn solidify(&mut self, pa: f64, pk: f64) {
        let decisions: Vec<(usize, bool)> = self
            .cells
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &s)| {
                if s != CellState::Diffusing {
                    return None;
                }
                let (x, y) = self.index_to_coords(idx);
                let prob = self.calculate_attachment_probability(x, y, pa, pk);
                Some((idx, prob > 0.0 && rand::rng().random_bool(prob)))
            })
            .collect();

        self.apply_solidification_decisions(&decisions);
    }

    /// Perform solidification step with periodic attachment probability potential
    /// pa(y) = pa_max * sin^2(k * 2pi * y / width)
    ///
    /// Scans cells directly and rebuilds `diffusing_indices`, like `solidify`.
    pub fn solidify_periodic(&mut self, pa_max: f64, pk: f64, pa_k: f64) {
        let decisions: Vec<(usize, bool)> = self
            .cells
            .par_iter()
            .enumerate()
            .filter_map(|(idx, &s)| {
                if s != CellState::Diffusing {
                    return None;
                }
                let (x, y) = self.index_to_coords(idx);
                let prob = self.calculate_periodic_attachment_probability(x, y, pa_max, pk, pa_k);
                Some((idx, prob > 0.0 && rand::rng().random_bool(prob)))
            })
            .collect();

        self.apply_solidification_decisions(&decisions);
    }

    fn apply_solidification_decisions(&mut self, decisions: &[(usize, bool)]) {
        let mut new_diffusing = Vec::with_capacity(decisions.len());

        for &(idx, should_solidify) in decisions {
            if should_solidify {
                self.cells[idx] = CellState::Solid;
                self.solid_indices.push(idx);
            } else {
                new_diffusing.push(idx);
            }
        }

        self.diffusing_indices = new_diffusing;
    }
}

#[cfg(test)]
mod tests;
