@group(0) @binding(0)
var<storage, read_write> output: array<atomic<u32>>;

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

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // var rand = hash32(global_id.x + u32(pc.time*1000));
	var rand = vec2u(hash32(global_id.x), hash32(global_id.y));
    var z = vec2f(0.0,0.0);
    var index = vec2u(0,0);
	var max_value = 0u;
    //todo generate random number

    for(var j: u32 = 0; j < pc.iters_per_invocation; j++){
        // rand = hash32(rand);
		var rand = vec2u(hash32(rand.x), hash32(rand.y));

        /*switch (j % 3) {
			case 0: {
				z = 0.5*z + 0.5*vec2f(cos(0.0),  sin(0.0));
			}
			case 1: {
				z = 0.5*z + 0.5*vec2f(cos(tau/3.0),  sin(tau/3.0));
			}
			case 2: {
				z = 0.5*z + 0.5*vec2f(cos(2.0*tau/3.0),  sin(2.0*tau/3.0));
			}
			default: {
				z = vec2f(0.5);
			}
        }   
        index = vec2u((z + vec2f(1.0))/2.0)*pc.width;*/
        

		let dx = perlinNoise2(vec2f(rand));
		let dy = perlinNoise2(-vec2f(rand));
		// let dx = f32(global_id.x) / f32(pc.width);
		// let dy = f32(global_id.y) / f32(pc.height);
		let fpos = ((vec2f(dx, dy) + 0.0) / 1.0) * vec2f(pc.width);
		index = vec2u(u32(fpos.x), u32(fpos.y));
        if ((index.x < pc.width) && (index.y < pc.height)){
			var old = atomicAdd(&output[(index.x) + (index.y) * pc.width], 1u);
			max_value = max(old + 1u, max_value);
        }
    }
	atomicMax(&metadata.max, max_value);

	// let index = global_id.y * pc.width + global_id.x;
	// atomicStore(&output[index], rand % 10u);
	// atomicStore(&metadata.max, 10u);
}

fn permute4(x: vec4f) -> vec4f { return ((x * 34. + 1.) * x) % vec4f(289.); }
fn fade2(t: vec2f) -> vec2f { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise2(P: vec2f) -> f32 {
    var Pi: vec4f = floor(P.xyxy) + vec4f(0., 0., 1., 1.);
    let Pf = fract(P.xyxy) - vec4f(0., 0., 1., 1.);
    Pi = Pi % vec4f(289.); // To avoid truncation effects in permutation
    let ix = Pi.xzxz;
    let iy = Pi.yyww;
    let fx = Pf.xzxz;
    let fy = Pf.yyww;
    let i = permute4(permute4(ix) + iy);
    var gx: vec4f = 2. * fract(i * 0.0243902439) - 1.; // 1/41 = 0.024...
    let gy = abs(gx) - 0.5;
    let tx = floor(gx + 0.5);
    gx = gx - tx;
    var g00: vec2f = vec2f(gx.x, gy.x);
    var g10: vec2f = vec2f(gx.y, gy.y);
    var g01: vec2f = vec2f(gx.z, gy.z);
    var g11: vec2f = vec2f(gx.w, gy.w);
    let norm = 1.79284291400159 - 0.85373472095314 *
        vec4f(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 = g00 * norm.x;
    g01 = g01 * norm.y;
    g10 = g10 * norm.z;
    g11 = g11 * norm.w;
    let n00 = dot(g00, vec2f(fx.x, fy.x));
    let n10 = dot(g10, vec2f(fx.y, fy.y));
    let n01 = dot(g01, vec2f(fx.z, fy.z));
    let n11 = dot(g11, vec2f(fx.w, fy.w));
    let fade_xy = fade2(Pf.xy);
    let n_x = mix(vec2f(n00, n01), vec2f(n10, n11), vec2f(fade_xy.x));
    let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}