use crate::attachment_engine::CellState;
use crate::simulation::{PaState, SimulationState};
use eframe;
use eframe::egui;
use egui::{Color32, ColorImage, Response, Sense, TextureHandle, TextureOptions, Vec2};
use rayon::prelude::*;

const EMPTY_COLOR: Color32 = Color32::BLACK;
const DIFFUSING_COLOR: Color32 = Color32::from_rgb(100, 100, 255);
const SOLID_COLOR: Color32 = Color32::from_rgb(200, 200, 200);
const BORDER_COLOR: Color32 = Color32::from_rgb(128, 0, 128);
const SINUSOID_COLOR: Color32 = Color32::GREEN;
const FONT_SIZE: f32 = 15.0;
pub struct Gui {
    simulation_state: SimulationState,
    last_frame_time: std::time::Instant,
    grid_texture: Option<TextureHandle>,
    fps: f64,
    target_fps: f64,
    started: bool,
    paused: bool,
    log_sliders: bool,
    show_help: bool,
}

impl Gui {
    pub fn new() -> Self {
        Self {
            simulation_state: SimulationState::new(
                100,
                100,
                0.1,
                1,
                PaState::Constant,
                0.001,
                1.0,
                1.0,
            ),
            last_frame_time: std::time::Instant::now(),
            grid_texture: None,
            fps: 0.0,
            target_fps: 60.0,
            started: false,
            paused: false,
            log_sliders: false,
            show_help: true,
        }
    }

    fn update_grid_texture(&mut self, ctx: &egui::Context) {
        let grid = self.simulation_state.get_grid_view();
        let width = grid.width();
        let height = grid.height();

        let pixels: Vec<u8> = (0..width * height)
            .into_par_iter()
            .flat_map(|i| {
                let x = i / height;
                let y = i % height;
                let color = match grid[(x, y)] {
                    CellState::Empty => EMPTY_COLOR,
                    CellState::Diffusing => DIFFUSING_COLOR,
                    CellState::Solid => SOLID_COLOR,
                };
                [color.r(), color.g(), color.b(), 255]
            })
            .collect();

        let image = ColorImage::from_rgba_unmultiplied([width, height], &pixels);

        // Use nearest neighbor filtering for crisp pixels
        let texture_options = TextureOptions {
            magnification: egui::TextureFilter::Nearest,
            minification: egui::TextureFilter::Nearest,
            wrap_mode: egui::TextureWrapMode::ClampToEdge,
            mipmap_mode: None,
        };

        if let Some(texture) = &mut self.grid_texture {
            texture.set(image, texture_options);
        } else {
            let texture = ctx.load_texture("grid_texture", image, texture_options);
            self.grid_texture = Some(texture);
        }
    }

    fn draw_grid(&mut self, ui: &mut egui::Ui) -> Response {
        let texture = self.grid_texture.as_ref().unwrap();
        let available_size = ui.available_size();

        let grid = &self.simulation_state.get_grid_view();
        let grid_aspect = grid.width() as f32 / grid.height() as f32;
        let available_aspect = available_size.x / available_size.y;

        let size = if available_aspect > grid_aspect {
            Vec2::new(available_size.y * grid_aspect, available_size.y)
        } else {
            Vec2::new(available_size.x, available_size.x / grid_aspect)
        };

        // Center the grid
        let available_rect = ui.available_rect_before_wrap();
        let centered_rect = egui::Rect::from_center_size(available_rect.center(), size);

        let response = ui.allocate_rect(centered_rect, Sense::hover());

        let mut mesh = egui::epaint::Mesh::with_texture(texture.id());
        mesh.add_rect_with_uv(
            centered_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        ui.painter().add(mesh);

        // Draw purple border
        ui.painter().rect_stroke(
            centered_rect,
            0.0,
            egui::Stroke::new(3.0, BORDER_COLOR),
            egui::StrokeKind::Outside,
        );

        // draw an upside down sinuosoidal path at the top of the rectangle if in periodic mode
        // y = pa_max * sin^2(k * 2π * x / width)
        if let PaState::Periodic = self.simulation_state.pa_state {
            let points: Vec<egui::Pos2> = (0..=100)
                .map(|i| {
                    let t = i as f32 / 100.0;
                    let x = centered_rect.left() + t * centered_rect.width();
                    let frequency = self.simulation_state.pa_wavenumber as f32;
                    let amplitude = 30.0;
                    let y = centered_rect.top()
                        + self.simulation_state.pa as f32
                            * (amplitude
                                * (frequency * 2.0 * std::f32::consts::PI * t).sin().powi(2));
                    egui::pos2(x, y)
                })
                .collect();

            ui.painter().add(egui::epaint::Shape::line(
                points,
                egui::Stroke::new(5.0, SINUSOID_COLOR),
            ));
        }

        response
    }

    fn draw_statistics(&mut self, ui: &mut egui::Ui) {
        // Display number of solid atoms, diffusing atoms, total atoms, and FPS
        let grid = self.simulation_state.get_grid_view();
        let solid_count = grid.solid_count();
        let diffusing_count = grid.diffusing_count();
        let total_count = solid_count + diffusing_count;
        ui.label(egui::RichText::new(format!("Solid Atoms: {}", solid_count)).size(FONT_SIZE));
        ui.label(
            egui::RichText::new(format!("Diffusing Atoms: {}", diffusing_count)).size(FONT_SIZE),
        );
        ui.label(egui::RichText::new(format!("Total Atoms: {}", total_count)).size(FONT_SIZE));
        ui.label(egui::RichText::new(format!("FPS: {:.1}", self.fps)).size(FONT_SIZE));
    }

    fn draw_simulation_control_btns(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let start_text = if self.started { "Restart" } else { "Start" };
            if ui
                .button(egui::RichText::new(start_text).size(FONT_SIZE))
                .clicked()
            {
                self.started = true;
                self.paused = false;
                self.simulation_state = SimulationState::new(
                    self.simulation_state.get_grid_view().width(),
                    self.simulation_state.get_grid_view().height(),
                    self.simulation_state.c0,
                    self.simulation_state.nds,
                    self.simulation_state.pa_state,
                    self.simulation_state.pa,
                    self.simulation_state.pk,
                    self.simulation_state.pa_wavenumber,
                );
            }
            let pause_text = if self.paused { "Resume" } else { "Pause" };
            if ui
                .button(egui::RichText::new(pause_text).size(FONT_SIZE))
                .clicked()
            {
                if self.started {
                    self.paused = !self.paused;
                }
            };
        });

        let pa_mode_text = match self.simulation_state.pa_state {
            PaState::Constant => "Pa mode: Constant",
            PaState::Periodic => "Pa mode: Periodic",
        };
        if ui
            .button(egui::RichText::new(pa_mode_text).size(FONT_SIZE))
            .clicked()
        {
            self.simulation_state.pa_state = match self.simulation_state.pa_state {
                PaState::Constant => PaState::Periodic,
                PaState::Periodic => PaState::Constant,
            };
        }
    }

    fn draw_simulation_parameters_widgets(&mut self, ui: &mut egui::Ui) {
        ui.heading("Simulation parameters");
        ui.separator();
        ui.add_space(10.0);
        ui.add(
            egui::Slider::new(&mut self.simulation_state.c0, 0.0..=1.0)
                .text(egui::RichText::new("Initial Concentration (c₀)").size(FONT_SIZE))
                .logarithmic(self.log_sliders),
        );
        ui.add(
            egui::Slider::new(&mut self.simulation_state.nds, 1..=300)
                .text(egui::RichText::new("Number of diffusion steps (nds)").size(FONT_SIZE)),
        );
        ui.add(
            egui::Slider::new(&mut self.simulation_state.pk, 0.0..=1.0)
                .text(egui::RichText::new("A2K probability (Pk)").size(FONT_SIZE))
                .logarithmic(self.log_sliders),
        );

        match self.simulation_state.pa_state {
            PaState::Constant => {
                ui.add(
                    egui::Slider::new(&mut self.simulation_state.pa, 1e-6..=1.0)
                        .text(egui::RichText::new("Constant A21 Probability (Pa)").size(FONT_SIZE))
                        .logarithmic(self.log_sliders),
                );
            }
            PaState::Periodic => {
                ui.add(
                    egui::Slider::new(&mut self.simulation_state.pa, 1e-6..=1.0)
                        .text(
                            egui::RichText::new("Amplitude of A21 Probability (Pa)")
                                .size(FONT_SIZE),
                        )
                        .logarithmic(self.log_sliders),
                );
                ui.add(
                    egui::Slider::new(&mut self.simulation_state.pa_wavenumber, 0.1..=10.0)
                        .text(egui::RichText::new("Wavenumber of Pa modulation").size(FONT_SIZE))
                        .logarithmic(self.log_sliders),
                );
            }
        }

        ui.checkbox(&mut self.log_sliders, "Logarithmic Sliders");
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        self.draw_simulation_parameters_widgets(ui);

        ui.add_space(20.0);

        self.draw_simulation_control_btns(ui);

        ui.add_space(20.0);
        ui.heading("Statistics");
        ui.separator();
        ui.add_space(10.0);
        self.draw_statistics(ui);

        // Fill remaining space and place Help button at bottom right
        ui.with_layout(egui::Layout::bottom_up(egui::Align::RIGHT), |ui| {
            let button = egui::Button::new(egui::RichText::new("  Help  ").size(18.0));
            if ui.add(button).clicked() {
                self.show_help = true;
                self.paused = true;
            }
        });
    }

    fn draw_splash_screen(&mut self, ctx: &egui::Context) {
        egui::Window::new("Crystal Step Life 2D")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Welcome to Crystal Step Life 2D");
                    ui.add_space(10.0);
                });

                ui.separator();
                ui.add_space(10.0);

                ui.label(egui::RichText::new("Controls:").size(FONT_SIZE).strong());
                ui.add_space(5.0);

                egui::Grid::new("help_grid")
                    .num_columns(2)
                    .spacing([20.0, 8.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("c₀").size(FONT_SIZE).strong());
                        ui.label(
                            egui::RichText::new("Initial concentration of diffusing atoms")
                                .size(FONT_SIZE),
                        );
                        ui.end_row();

                        ui.label(egui::RichText::new("nds").size(FONT_SIZE).strong());
                        ui.label(
                            egui::RichText::new("Number of diffusion steps per simulation step")
                                .size(FONT_SIZE),
                        );
                        ui.end_row();

                        ui.label(egui::RichText::new("Pa").size(FONT_SIZE).strong());
                        ui.label(
                            egui::RichText::new("Probability of attachment (A21 transition)")
                                .size(FONT_SIZE),
                        );
                        ui.end_row();

                        ui.label(egui::RichText::new("Pa mode").size(FONT_SIZE).strong());
                        ui.label(
                            egui::RichText::new("Constant or periodic spatial modulation of Pa")
                                .size(FONT_SIZE),
                        );
                        ui.end_row();

                        ui.label(egui::RichText::new("Wavenumber").size(FONT_SIZE).strong());
                        ui.label(
                            egui::RichText::new("Frequency of Pa modulation (periodic mode)")
                                .size(FONT_SIZE),
                        );
                        ui.end_row();
                    });

                ui.add_space(15.0);
                ui.separator();
                ui.add_space(10.0);

                ui.label(egui::RichText::new("Cell Colors:").size(FONT_SIZE).strong());
                ui.add_space(5.0);

                egui::Grid::new("color_grid")
                    .num_columns(2)
                    .spacing([20.0, 8.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("■").size(FONT_SIZE).color(SOLID_COLOR));
                        ui.label(egui::RichText::new("Solid (crystallized) atoms").size(FONT_SIZE));
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("■")
                                .size(FONT_SIZE)
                                .color(DIFFUSING_COLOR),
                        );
                        ui.label(egui::RichText::new("Diffusing atoms").size(FONT_SIZE));
                        ui.end_row();

                        ui.label(
                            egui::RichText::new("■")
                                .size(FONT_SIZE)
                                .color(EMPTY_COLOR)
                                .background_color(Color32::GRAY),
                        );
                        ui.label(egui::RichText::new("Empty cells").size(FONT_SIZE));
                        ui.end_row();
                    });

                ui.add_space(20.0);

                ui.separator();
                ui.add_space(10.0);

                ui.label(egui::RichText::new("In detail:").size(FONT_SIZE).strong());
                ui.add_space(5.0);

                // TODO: Finish help text
                ui.label(
                    egui::RichText::new(
                        "This simulation is based on the following attachment events to an initial\
                solid crystal wall at the bottom of the grid.",
                    )
                    .size(FONT_SIZE),
                );

                ui.add_space(20.0);

                ui.vertical_centered(|ui| {
                    if ui
                        .button(egui::RichText::new("Close").size(FONT_SIZE))
                        .clicked()
                    {
                        self.show_help = false;
                    }
                });
            });
    }
}

impl eframe::App for Gui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        self.fps = if dt > 0.0 { 1.0 / dt as f64 } else { 0.0 };

        if self.fps > self.target_fps {
            std::thread::sleep(std::time::Duration::from_secs_f32(
                (1.0 / self.target_fps as f32) - dt,
            ));
        }

        if self.started && !self.paused {
            self.simulation_state.step();
        }

        self.update_grid_texture(ctx);
        egui::SidePanel::right("controls")
            .min_width(300.0)
            .default_width(400.0)
            .show(ctx, |ui| {
                self.draw_controls(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_grid(ui);
        });

        if self.show_help {
            self.draw_splash_screen(ctx);
        }

        ctx.request_repaint()
    }
}
