
struct TransformData{
    scale: f32,
    rotation: f32,
    translation_x: f32,
    translation_y: f32, 
}


struct RenderedData{
    histogram: Vec<u32>,
    max: u32,
}

struct Metadata{
    width: u32,
    height:u32,
    iterations_per_invocation:u32,
}

@group(0) @binding(0)
var<storage, read_write> output: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> entropy: array<u32>;

@group(0) @binding(1)
var<storage, read> metadata: Metadata;

@group(1) @binding(0)
var<storage, read_write> metadata: Metadata;

struct PushConstants {
	width: u32,
	height: u32,
	time: f32,
    iters_per_invocation: u32,
}
var<push_constant> pc: PushConstants;


//constants
const pi: f32 = 3.141592653589793;
const tau: f32 = 6.283185307179586;

fn hash32(n: u32) -> u32 {
    var h32 = n + 374761393u;
    h32 = 668265263u * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = 2246822519u * (h32 ^ (h32 >> 15));
    h32 = 3266489917u * (h32 ^ (h32 >> 13));
    return h32^(h32 >> 16);
}



@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    
    var z = vec2f(0.0,0.0);
    var index = vec2u(0,0);
	var max_value = 0u;

    for(var j: u32 = 0; j < pc.iters_per_invocation; j++){
        
        let rand = entropy[j] ^ hash32(global_id.x);
        
        switch u32(rand % 3) {
			case 0: {
				z = (1-ratio)*z + ratio*vec2f(cos(0.0),  sin(0.0));
			}
			case 1: {
				z = (1-ratio)*z + ratio*vec2f(cos(tau/3.0),  sin(tau/3.0));
			}
			case 2: {
				z = (1-ratio)*z + ratio*vec2f(cos(2.0*tau/3.0),  sin(2.0*tau/3.0));
			}
			default: {
				z = vec2f(0.0);
			}
        }
        index = vec2u(((z + 1.0) / 2.0) * f32(box_size));
   
        if ((index.x < pc.width) && (index.y < pc.height) && j >10){
			var old = atomicAdd(&output[(index.x) + (index.y) * pc.width], 1u);
			max_value = max(old + 1u, max_value);
        }
    }
	atomicMax(&metadata.max, max_value);
}


