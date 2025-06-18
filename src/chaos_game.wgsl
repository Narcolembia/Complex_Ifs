@group(0) @binding(0)
var<storage, read_write> output: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> entropy: array<u32>;


struct Metadata {
	max: atomic<u32>,
}

@group(1) @binding(0)
var<storage, read_write> metadata: Metadata;

struct PushConstants {
	width: u32,
	height: u32,
	time: f32,
    iters_per_invocation: u32,
}
var<push_constant> pc: PushConstants;


/*fn c_mult(lhs: vec2f, rhs: vec2f){
    return vec2f(lhs.x*rhs.x - lhs.y*lhs.y,lhs.x*rhs.y + lhs.y*rhs.x);
}

fn c_div(vec2f lhs, vec2f rhs){

}
fn c_exp(vec2f lhs, vec2f rhs){

}
*/

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

fn hash32_2d(p: vec2u) -> u32 {
    let p2 = 2246822519u; let p3 = 3266489917u;
    let p4 = 668265263u; let p5 = 374761393u;
    var h32 = p.y + p5 + p.x * p3;
    h32 = p4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = p2 * (h32^(h32 >> 15));
    h32 = p3 * (h32^(h32 >> 13));
    return h32^(h32 >> 16);
}

fn randrange_unbiased(seed: u32, max: u32) -> u32 {
    var seed_h = seed;
    let limit = 0xFFFFFFFF - 0xFFFFFFFF % max;
    loop {
        if seed_h >= limit {
            break;
        }
        seed_h = hash32(seed_h);
    }
    return seed_h  % max;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // var rand = hash32_2d(global_id.xy + u32(pc.time*1000));
	// var rand = vec2u(hash32(global_id.x), hash32(global_id.y));
    //var seed = hash32(u32(pc.time * 1000) ^ hash32_2d(global_id.x)) & 0xFFFF;
    //var rand = vec2f(global_id.xy + seed);
    let box_size: u32 = min(pc.width,pc.height);
    //var z = vec2f( f32(entropy[0] ^ hash32(global_id.x)) ,f32(entropy[1] ^ hash32(global_id.x)));
    //z = (2.0*z/length(z)) - 1;
    var z = vec2f(0.0,0.0);
    var index = vec2u(0,0);
	var max_value = 0u;
    let seed = global_id.x ^ u32(pc.time*1000);
    //todo generate random number

    for(var j: u32 = 0; j < pc.iters_per_invocation; j++){
    
        let rand = entropy[j] ^ hash32(seed) ;
        let ratio :f32 = 0.6;
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


