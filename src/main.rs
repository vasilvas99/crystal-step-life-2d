#![windows_subsystem = "windows"] // hide console window on Windows
pub mod attachment_engine;
mod gui;
pub mod simulation;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_resizable(true),
        ..Default::default()
    };

    eframe::run_native(
        "Crystal Growth Simulation",
        options,
        Box::new(|_cc| Ok(Box::new(gui::Gui::new()))),
    )
}
