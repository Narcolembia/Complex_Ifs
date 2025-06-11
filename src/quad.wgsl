struct VertexOut {
	@builtin(position)
	position: vec4<f32>,

	@location(0)
	uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
	var position: vec4<f32>;
	var uv: vec2<f32>;
	switch index {
		case 0, 3: {
			position = vec4<f32>(-1.0, 1.0, 0.0, 1.0);
			uv = vec2<f32>(0.0, 1.0);
		}
		case 1: {
			position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
			uv = vec2<f32>(0.0, 0.0);
		}
		case 2, 4: {
			position = vec4<f32>(1.0, -1.0, 0.0, 1.0);
			uv = vec2<f32>(1.0, 0.0);
		}
		case 5: {
			position = vec4<f32>(1.0, 1.0, 0.0, 1.0);
			uv = vec2<f32>(1.0, 1.0);
		}
		default: {
			position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
			uv = vec2<f32>(0.0, 0.0);
		}
	}
	return VertexOut(position, uv);
}

@group(0)
@binding(0)
var<storage, read_write> compute_output: array<atomic<u32>>;

struct Metadata {
	max: atomic<u32>,
}

@group(1) @binding(0)
var<storage, read_write> metadata: Metadata;

struct PushConstants {
	width: u32,
	height: u32,
	time: f32,
}
var<push_constant> pc: PushConstants;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
	let px = in.uv.x * f32(pc.width);
	let py = in.uv.y * f32(pc.height);
	let index = u32(py) * pc.width + u32(px);
	// let int_color = atomicLoad(&compute_output[index]);
	// let r = (int_color & 0xff0000) >> 16;
	// let g = (int_color & 0x00ff00) >> 8;
	// let b = (int_color & 0x0000ff) >> 0;
	// let color = vec3<f32>(f32(r) / 255.0, f32(g) / 255.0, f32(b) / 255.0);
	let color = vec3<f32>(f32(atomicLoad(&compute_output[index])) / f32(atomicLoad(&metadata.max)));
	return vec4<f32>(color, 1.0);
	// return vec4<f32>(in.uv, 0.0, 1.0);
}