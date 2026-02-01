use super::*;

#[test]
fn test_grid_initialization() {
    let grid = Grid::new(100, 100, 0.5);
    assert_eq!(grid.width, 100);
    assert_eq!(grid.height, 100);
    assert_eq!(grid.cells.len(), 10000);

    // Check that bottom row is solid (x=99, all y columns)
    for y in 0..100 {
        assert_eq!(grid.get_cell(99, y), &CellState::Solid);
    }

    // Check that solid_indices contains bottom row
    assert_eq!(grid.solid_indices.len(), 100);
}

#[test]
fn test_grid_concentration_initialization() {
    let grid = Grid::new(100, 100, 0.3);

    let empty_count = grid
        .cells
        .iter()
        .filter(|&&s| s == CellState::Empty)
        .count();
    let diffusing_count = grid.diffusing_indices.len();
    let solid_count = grid.solid_indices.len();

    // Total should equal grid size
    assert_eq!(empty_count + diffusing_count + solid_count, 10000);

    // Concentration should be approximately 0.3
    let total_available = empty_count + diffusing_count;
    let actual_c0 = diffusing_count as f64 / total_available as f64;
    assert!(
        (actual_c0 - 0.3).abs() < 0.01,
        "Concentration mismatch: expected ~0.3, got {}",
        actual_c0
    );
}

#[test]
fn test_index_to_coords_conversion() {
    let grid = Grid::new(100, 100, 0.0);

    // Test various positions (x=row, y=column, index=x*width+y)
    let test_cases = vec![
        (0, 0, 0),
        (0, 5, 5),
        (1, 0, 100),
        (50, 50, 5050),
        (99, 99, 9999),
    ];

    for (x, y, expected_index) in test_cases {
        let index = grid.get_index(x, y);
        assert_eq!(index, expected_index, "get_index({}, {}) failed", x, y);

        let (restored_x, restored_y) = grid.index_to_coords(index);
        assert_eq!(
            restored_x, x,
            "index_to_coords x failed for index {}",
            index
        );
        assert_eq!(
            restored_y, y,
            "index_to_coords y failed for index {}",
            index
        );
    }
}

#[test]
fn test_diffuse_step_preserves_atom_count() {
    let mut grid = Grid::new(50, 50, 0.3);

    let initial_diffusing = grid.diffusing_indices.len();
    let initial_solid = grid.solid_indices.len();
    let initial_empty = grid
        .cells
        .iter()
        .filter(|&&s| s == CellState::Empty)
        .count();

    // Perform diffusion step
    grid.diffuse();

    let final_diffusing = grid.diffusing_indices.len();
    let final_solid = grid.solid_indices.len();
    let final_empty = grid
        .cells
        .iter()
        .filter(|&&s| s == CellState::Empty)
        .count();

    // Atom counts should be preserved
    assert_eq!(
        final_diffusing, initial_diffusing,
        "Diffusing count changed"
    );
    assert_eq!(final_solid, initial_solid, "Solid count changed");
    assert_eq!(final_empty, initial_empty, "Empty count changed");

    // Verify diffusing_indices matches actual grid state
    let actual_diffusing_on_grid = grid
        .cells
        .iter()
        .filter(|&&s| s == CellState::Diffusing)
        .count();
    assert_eq!(
        actual_diffusing_on_grid, final_diffusing,
        "Diffusing indices out of sync with grid"
    );
}

#[test]
fn test_diffuse_step_no_atoms_stuck_in_solid() {
    let mut grid = Grid::new(50, 50, 0.3);

    // Perform multiple diffusion steps
    for _ in 0..10 {
        grid.diffuse();

        // Check that no diffusing atoms are in solid positions
        for &idx in &grid.diffusing_indices {
            assert_eq!(
                grid.get_cell_by_index(idx),
                CellState::Diffusing,
                "Diffusing atom index points to non-diffusing cell"
            );
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
    assert_eq!(
        grid.diffusing_indices.len(),
        initial_diffusing,
        "Isolated atom solidified without neighbors"
    );
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
    assert_eq!(
        final_total, initial_total,
        "Total atoms not preserved in periodic solidification"
    );
}

#[test]
fn test_check_neighbors_detects_kink_sites() {
    let mut grid = Grid::new(50, 50, 0.0);

    // Create a kink site pattern (x=row, y=column):
    // The bottom row (x=49) is already solid
    // Add solids to create a kink above it:
    // Row 47: . X .  <- test position (47, 24) should detect kink
    // Row 48: X X .  <- (48, 24) and (48, 25) are solid
    // Row 49: S S S  <- bottom row (already all solid)

    let test_row = 48; // One row above bottom (bottom is 49)
    let test_col = 25;

    // Add a solid at (test_row, test_col)
    grid.set_cell(test_row, test_col, CellState::Solid);

    // Position at (test_row-1, test_col) should be a kink site:
    // - below: (test_row, test_col) = solid ✓
    // - below: (test_row+1, test_col) = also solid (bottom row) ✓
    // Wait, that's not right. Let me reconsider...

    // Actually, for position (47, 25):
    // - below: (48, 25) = solid ✓
    // - right: (47, 26) = empty
    // - left: (47, 24) = need to make solid for kink
    grid.set_cell(test_row - 1, test_col - 1, CellState::Solid);

    // Position at (test_row-1, test_col) should now be a kink:
    // - below (test_row, test_col) = solid ✓
    // - left (test_row-1, test_col-1) = solid ✓
    // kink = below && (left || right) = true
    let (is_kink, has_neighbor) = grid.check_neighbors(test_row - 1, test_col);

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
    assert!(
        final_diffusing > initial_diffusing,
        "Failed to add atoms when increasing concentration"
    );
}

#[test]
fn test_adjust_concentration_removes_atoms() {
    let mut grid = Grid::new(50, 50, 0.5);

    let initial_diffusing = grid.diffusing_indices.len();

    // Decrease target concentration
    grid.adjust_diffusing_concentration(0.2, 0.5);

    let final_diffusing = grid.diffusing_indices.len();

    // Should have removed atoms
    assert!(
        final_diffusing < initial_diffusing,
        "Failed to remove atoms when decreasing concentration"
    );
}

#[test]
fn test_find_topmost_solid_row() {
    let mut grid = Grid::new(50, 50, 0.0);

    // Initially only bottom row is solid (row 49)
    let topmost = grid.find_topmost_solid_row();
    assert_eq!(topmost, 49, "Initial topmost should be bottom row");

    // Add solid at row 30, column 25 (x=30, y=25)
    grid.set_cell(30, 25, CellState::Solid);
    grid.solid_indices.push(grid.get_index(30, 25));

    let topmost = grid.find_topmost_solid_row();
    assert_eq!(topmost, 30, "Failed to find topmost solid at row 30");

    // Add solid at row 20, column 25 (x=20, y=25)
    grid.set_cell(20, 25, CellState::Solid);
    grid.solid_indices.push(grid.get_index(20, 25));

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
    assert_eq!(
        sorted_diffusing.len(),
        unique_count_before,
        "Duplicate indices in diffusing_indices"
    );

    // Check for duplicate indices in solid_indices
    let mut sorted_solid = grid.solid_indices.clone();
    sorted_solid.sort_unstable();
    let unique_solid_before = sorted_solid.len();
    sorted_solid.dedup();
    assert_eq!(
        sorted_solid.len(),
        unique_solid_before,
        "Duplicate indices in solid_indices"
    );

    // Check no overlap between diffusing and solid
    for &diff_idx in &grid.diffusing_indices {
        assert!(
            !grid.solid_indices.contains(&diff_idx),
            "Index {} is in both diffusing and solid lists",
            diff_idx
        );
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

        assert_eq!(
            diffusing_count_from_grid,
            grid.diffusing_indices.len(),
            "Diffusing count mismatch between grid and indices"
        );
        assert_eq!(
            solid_count_from_grid,
            grid.solid_indices.len(),
            "Solid count mismatch between grid and indices"
        );

        // Verify all indices point to correct cell states
        for &idx in &grid.diffusing_indices {
            assert_eq!(
                grid.cells[idx],
                CellState::Diffusing,
                "Diffusing index {} points to {:?}",
                idx,
                grid.cells[idx]
            );
        }

        for &idx in &grid.solid_indices {
            assert_eq!(
                grid.cells[idx],
                CellState::Solid,
                "Solid index {} points to {:?}",
                idx,
                grid.cells[idx]
            );
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
    assert_eq!(
        grid.diffusing_indices.len(),
        initial_diffusing,
        "Atoms solidified with zero probability"
    );
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

    // The bottom row should be x=99 (row 99), all y columns
    for y in 0..100 {
        let idx = grid.get_index(99, y);
        let (restored_x, restored_y) = grid.index_to_coords(idx);
        assert_eq!(restored_x, 99);
        assert_eq!(restored_y, y);
        assert_eq!(
            grid.cells[idx],
            CellState::Solid,
            "Bottom row should be solid"
        );
    }
}

#[test]
fn test_find_topmost_uses_correct_coordinate() {
    let mut grid = Grid::new(50, 50, 0.0);

    // Bottom row is x=49 (highest x value, row 49)
    // Add a solid at x=10, y=25 (row 10, which is "topmost" visually)
    grid.set_cell(10, 25, CellState::Solid);
    grid.solid_indices.push(grid.get_index(10, 25));

    let topmost = grid.find_topmost_solid_row();

    // Should return 10 (minimum x/row), not 49
    assert_eq!(
        topmost, 10,
        "find_topmost_solid_row should return minimum x coordinate (row)"
    );
}
