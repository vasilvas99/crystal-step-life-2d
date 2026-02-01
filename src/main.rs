pub mod attachment_engine;
pub mod simulation;

use attachment_engine::{CellState, Grid};
use simulation::{PaState, SimulationState};
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H");
    io::stdout().flush().unwrap();
}

fn render_grid(grid: &Grid, show_area: (usize, usize, usize, usize)) {
    let (start_x, start_y, width, height) = show_area;

    // Top border
    print!("┌");
    for _ in 0..width {
        print!("─");
    }
    println!("┐");

    // Grid content
    for x in start_x..start_x + height {
        print!("│");
        for y in start_y..start_y + width {
            if x >= grid.height || y >= grid.width {
                print!(" ");
                continue;
            }

            let cell = grid.get_cell(x, y);
            let ch = match cell {
                CellState::Empty => ' ',
                CellState::Diffusing => '·',
                CellState::Solid => '^',
            };
            print!("{}", ch);
        }
        println!("│");
    }

    // Bottom border
    print!("└");
    for _ in 0..width {
        print!("─");
    }
    println!("┘");
}

fn main() {
    clear_screen();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║      Crystal Growth Simulation - ASCII Visualization     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Simulation parameters
    let grid_size = 80;
    let c0 = 0.15;
    let nds = 1;
    let pa = 0.15;
    let pk = 1.0;
    let steps = 10000;
    let display_width = 60;
    let display_height = 30;

    println!("Parameters:");
    println!("  Grid size: {}x{}", grid_size, grid_size);
    println!("  C0 (concentration): {}", c0);
    println!("  NDS (diffusion steps): {}", nds);
    println!("  PA (attachment probability): {}", pa);
    println!("  PK (kink probability): {}", pk);
    println!("  Total steps: {}", steps);
    println!();
    println!("Legend: [█] Solid  [·] Diffusing  [ ] Empty");
    println!();
    println!("Press Ctrl+C to stop...");
    println!();

    thread::sleep(Duration::from_secs(2));

    // Initialize grid
    let grid = Grid::new(grid_size, grid_size, c0);

    // Create simulation
    let mut sim = SimulationState::new(grid, c0, nds, PaState::Constant, pa, pk, 0.1);

    // Run simulation with visualization
    for i in 0..steps {
        sim.step();

        // Update display every 5 steps
        if i % 5 == 0 {
            clear_screen();

            println!("╔═══════════════════════════════════════════════════════════╗");
            println!("║      Crystal Growth Simulation - ASCII Visualization     ║");
            println!("╚═══════════════════════════════════════════════════════════╝");
            println!();

            // Stats
            println!(
                "Step: {}/{}  │  Solid: {}  │  Diffusing: {}  │  Growth: {} cells",
                sim.get_time_step(),
                steps,
                sim.count_solid_cells(),
                sim.count_diffusing_cells(),
                sim.count_solid_cells() - grid_size
            );
            println!();

            // Calculate viewing window to show bottom of grid (where crystal grows)
            let grid_ref = sim.grid();
            let start_x = if grid_size > display_height {
                grid_size - display_height // Show bottom portion
            } else {
                0
            };
            let start_y = if grid_size > display_width {
                (grid_size - display_width) / 2 // Center horizontally
            } else {
                0
            };

            render_grid(grid_ref, (start_x, start_y, display_width, display_height));

            println!();
            println!(
                "Progress: [{}{}] {:.1}%",
                "█".repeat((i * 50) / steps),
                "░".repeat(50 - (i * 50) / steps),
                (i as f64 / steps as f64) * 100.0
            );

            thread::sleep(Duration::from_millis(100));
        }
    }

    // Final display
    clear_screen();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║            Simulation Complete - Final State             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    let grid_ref = sim.grid();
    let start_x = if grid_size > display_height {
        grid_size - display_height // Show bottom portion
    } else {
        0
    };
    let start_y = if grid_size > display_width {
        (grid_size - display_width) / 2 // Center horizontally
    } else {
        0
    };

    render_grid(grid_ref, (start_x, start_y, display_width, display_height));

    println!();
    println!("Final Statistics:");
    println!("  Total steps: {}", sim.get_time_step());
    println!("  Solid cells: {}", sim.count_solid_cells());
    println!("  Diffusing atoms: {}", sim.count_diffusing_cells());
    println!(
        "  Crystal growth: {} cells",
        sim.count_solid_cells() - grid_size
    );
    println!();
}
