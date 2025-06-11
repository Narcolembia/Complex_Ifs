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

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
	return vec4<f32>(in.uv, 0.0, 1.0);
}