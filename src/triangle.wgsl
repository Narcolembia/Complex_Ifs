struct VertexOut {
	@builtin(position)
	position: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
	var position: vec4f;
	switch index {
		case 0, 3: {
			position = vec4f(-0.75, 0.75, 0.0, 1.0);
		}
		case 1: {
			position = vec4f(-0.75, -0.75, 0.0, 1.0);
		}
		case 2, 4: {
			position = vec4f(0.75, -0.75, 0.0, 1.0);
		}
		case 5: {
			position = vec4f(1.0, 1.0, 0.0, 1.0);
		}
		default: {
			position = vec4f(0.0, 0.0, 0.0, 1.0);
		}
	}
	return VertexOut(position);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
	return vec4f(1.0, 0.0, 1.0, 1.0);
}
