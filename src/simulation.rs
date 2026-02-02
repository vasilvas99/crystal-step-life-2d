use crate::attachment_engine::{Grid, GridView};

pub enum PaState {
    Constant,
    Periodic,
}

pub struct SimulationState {
    time_step: usize,
    grid: Grid,
    c0_old: f64,
    c0: f64,
    nds: usize,
    pa_state: PaState,
    pa: f64,
    pk: f64,
    pa_wavenumber: f64,
}

impl SimulationState {
    pub fn new(
        grid: Grid,
        c0: f64,
        nds: usize,
        pa_state: PaState,
        pa: f64,
        pk: f64,
        pa_wavenumber: f64,
    ) -> Self {
        SimulationState {
            time_step: 0,
            grid,
            c0_old: c0,
            c0,
            nds,
            pa_state,
            pa,
            pk,
            pa_wavenumber,
        }
    }

    pub fn step(&mut self) {
        self.grid
            .adjust_diffusing_concentration(self.c0, self.c0_old);

        for _ in 0..self.nds {
            self.grid.diffuse();
        }

        match self.pa_state {
            PaState::Constant => {
                self.grid.solidify(self.pa, self.pk);
            }
            PaState::Periodic => {
                self.grid
                    .solidify_periodic(self.pa, self.pk, self.pa_wavenumber);
            }
        }

        self.time_step += 1;
    }

    pub fn get_grid_view(&self) -> GridView<'_> {
        self.grid.get_view()
    }

    pub fn get_time_step(&self) -> usize {
        self.time_step
    }

    pub fn count_solid_cells(&self) -> usize {
        self.grid.solid_indices.len()
    }

    pub fn count_diffusing_cells(&self) -> usize {
        self.grid.diffusing_indices.len()
    }

    pub fn set_pa_mode_constant(&mut self) {
        self.pa_state = PaState::Constant;
    }

    pub fn set_pa_mode_periodic(&mut self) {
        self.pa_state = PaState::Periodic;
    }

    pub fn update_c0(&mut self, new_c0: f64) {
        self.c0_old = self.c0;
        self.c0 = new_c0;
    }

    pub fn set_pa(&mut self, new_pa: f64) {
        self.pa = new_pa;
    }

    pub fn set_pk(&mut self, new_pk: f64) {
        self.pk = new_pk;
    }

    pub fn set_pa_wavenumber(&mut self, new_wavenumber: f64) {
        self.pa_wavenumber = new_wavenumber;
    }

    pub fn set_nds(&mut self, new_nds: usize) {
        self.nds = new_nds;
    }

    pub fn get_mode(&self) -> &PaState {
        &self.pa_state
    }

    pub fn get_c0(&self) -> f64 {
        self.c0
    }

    pub fn get_pa(&self) -> f64 {
        self.pa
    }

    pub fn get_pk(&self) -> f64 {
        self.pk
    }

    pub fn get_pa_wavenumber(&self) -> f64 {
        self.pa_wavenumber
    }

    pub fn get_nds(&self) -> usize {
        self.nds
    }
    
}
