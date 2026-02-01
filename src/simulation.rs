use crate::attachment_engine::Grid;

pub enum PaState {
    Constant,
    Periodic,
}

pub struct SimulationState {
    pub time_step: usize,
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
    pub fn new(grid: Grid, c0: f64, nds: usize, pa_state: PaState, pa: f64, pk: f64, pa_wavenumber: f64) -> Self {
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
        self.grid.adjust_diffusing_concentration(self.c0, self.c0_old);

        for i in 0..self.nds {
            self.grid.diffuse();
        }

        match self.pa_state {
            PaState::Constant => {
                self.grid.solidify(self.pa, self.pk);
            }
            PaState::Periodic => {
                self.grid.solidify_periodic(self.pa, self.pk, self.pa_wavenumber);
            }
        }
    }


}