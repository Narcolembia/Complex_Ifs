use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
pub struct Transform {
    rotation: f32,
    translation: (f32, f32),
    scale: f32,
}

#[derive(Default)]
pub struct RenderData {
    variables: Vec<(f32, f32)>,
    weights: Vec<f32>,
    transform: Transform,
}

pub type RenderDataRef = Rc<RefCell<RenderData>>;

#[derive(Default)]
pub struct RenderSettings {
    width: f32,
    height: f32,

    iters_per_invocation: u32,
    num_invocations: u32,
    num_passes: u32,

    weighted_random_size: u32,
}

pub fn default<T: Default>() -> T {
    T::default()
}
