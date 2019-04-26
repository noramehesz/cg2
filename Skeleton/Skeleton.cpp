//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

const float epsilon = 0.0001f;
GPUProgram gpuProgram; // vertex and fragment shaders
int kaleidoscope = 4;

struct Material {
public:
	bool isReflective;
	vec3 ambient;
	vec3 specular;
	vec3 diffuse;

	vec3 n;
	vec3 k;

	float shininess;

	Material() {}

	Material(vec3 ka, vec3 ks, vec3 kd, vec3 _n, vec3 _k, float shine, bool ir) {
		this->ambient = ka;
		this->specular = ks;
		this->diffuse = kd;
		this->n = _n;
		this->k = _k;
		this->isReflective = ir;
		this->shininess = shine;
	}

	vec3 reflect(vec3 direction, vec3 normal) {
		return direction - normal * dot(normal, direction) * 2.0f;
	}

	vec3 MaterialF0() {
		return vec3(
			(pow(n.x - 1.0f, 2.0f) + pow(k.x, 2.0f)) / (pow(n.x + 1.0f, 2.0f) + pow(k.x, 2.0f)),
			(pow(n.y - 1.0f, 2.0f) + pow(k.y, 2.0f)) / (pow(n.y + 1.0f, 2.0f) + pow(k.y, 2.0f)),
			(pow(n.z - 1.0f, 2.0f) + pow(k.z, 2.0f)) / (pow(n.z + 1.0f, 2.0f) + pow(k.z, 2.0f))
		);
	}

	vec3 Fresnel(vec3 direction, vec3 normal) {
		float cosTheta = fabs(dot(normal, direction));
		return MaterialF0() + (vec3(1, 1, 1) - MaterialF0()) * pow(1.0f - cosTheta, 5);
	}

};


struct Hit {
public:
	float hitDistance;
	vec3 hitPoint, hitNormal;
	Material* material;
	//Hit() {}
	Hit() { hitDistance = -1.0f; }
};

struct Light {
public:
	vec3 direction;
	vec3 energy;
	vec3 position;

	Light() {}

	Light(vec3 _direction, vec3 _energy) {
		this->direction = normalize(_direction);
		this->energy = _energy;
	}
};

struct Ray {
public:
	vec3 start;
	vec3 direction;
	Ray() {}
	Ray(vec3 _start, vec3 _dir) {
		this->start = _start;
		this->direction = _dir;
	}

};

class Intersectable {
public:
	Material * material;

	virtual Hit intersect(const Ray& ray) = 0;
};


struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		this->center = _center;
		this->radius = _radius;
		this->material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 distance = ray.start - center;

		float a = dot(ray.direction, ray.direction);
		float b = dot(distance, ray.direction) * 2.0f;
		float c = dot(distance, distance) - pow(radius, 2.0f);

		float discr = b * b - 4.0f * a * c;

		if (discr < 0) return hit;

		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / (2.0f * a);
		float t2 = (-b - sqrt_discr) / (2.0f * a);

		if (t1 < 0) return hit;

		hit.hitDistance = (t2 > 0) ? t2 : t1;
		hit.hitPoint = ray.start + ray.direction * hit.hitDistance;
		hit.hitNormal = normalize((hit.hitPoint - center) * (1.0f / radius));    //normalize((hit.hitPoint - center) * 2.0f); 
		hit.material = material;

		return hit;
	}

	float MVPtransf[4][4] = { {1, 0, 0, 0}, {0, 1, 0, 0 }, {0, 0, 1, 0},  {0, 0, 0, 1} };
 

	void setMateril(Material * mat) {
		this->material = mat;
	}

};


struct Plane : public Intersectable {
	vec3 point, normal;

	Plane(const vec3& _point, const vec3& _normal, Material* mat) {
		point = _point;
		normal = normalize(_normal);
		this->material = mat;
	}

	Hit intersect (const Ray& ray){
		Hit hit;
		double NdotV = dot(normal, ray.direction);
		if (fabs(NdotV) < epsilon) {
			return hit;
		}

		double t = dot(normal, point - ray.start) / NdotV;

		if (t < epsilon)
		{
			return hit;
		}

		hit.hitDistance = t;
		hit.hitPoint = ray.start + ray.direction * hit.hitDistance;
		hit.hitNormal = normal;

		if (dot(hit.hitNormal, ray.direction) > 0) {
			hit.hitNormal = hit.hitNormal * -1.0f;
		}

		hit.material = material;
		return hit;
	}

	void setMat(Material* mat) {
		this->material = mat; printf("set is done");
	}

};

class Camera {
public:
	vec3 eye;
	vec3 lookat;
	vec3 right;
	vec3 up;
	float XM;
	float YM;

	Camera(){
		
	}

	void set(vec3 _eye, vec3 _lookat, vec3 vup, double fov) {
		this->eye = _eye;
		this->lookat = _lookat;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2.0f);
		up = normalize(cross(w, right)) * f * tan(fov / 2.0f);

		XM = windowWidth;
		YM = windowHeight;
	}

	Ray getRay(int x, int y) {
		float alpha = (2.0f * (x+0.5f) / XM - 1);  
		float beta = (2.0f * (y+0.5f) / YM - 1);	
		vec3 dir = lookat + right * alpha + up * beta - eye;
		return Ray(eye, dir);
	}
};



class Scene {
public:
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	
	Camera camera;
	vec3 Lambient = vec3(0.4f, 0.4f, 0.4f);

	int maxdepth = 10;

	void addObject(Intersectable * obj) {
		objects.push_back(obj);
	}

	void addLight(Light * l) {
		lights.push_back(l);
	}

	void calculateKaleidoskop() {
		for (int i = 0; i < kaleidoscope; i++) {
			vec3 point;
			point.x =(float) cosf((((360 / kaleidoscope)*i) * 2 * M_PI) / 360) * 0.5f; //printf("%f   ", point.x);
			point.y =(float) sinf((((360 / kaleidoscope)*i) * 2 * M_PI) / 360) * 0.5f; //printf("%f   \n", point.y);
			point.z = 0.0f;
			Material ObjMaterial2(vec3(0.8, 0.6, 0.4), vec3(1.0, 0.5, 0.5), vec3(0.1, 0.8, 0.3), vec3(0, 0, 0), vec3(0, 0, 0), 10.0f, false);
			Sphere s(point, 0.05f, &ObjMaterial2); printf("%f   ", s.center.x); printf("%f   \n", s.center.y);

			//int num = objects.size();
			//objects[num+1] = &s;
			objects.push_back(new Sphere(point, 0.05f, &ObjMaterial2));
		}
	}

	void build(){  //ebbe kell a kaleidoszkop felepitese 
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		calculateKaleidoskop();
	}


	

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for(Intersectable * object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.hitDistance > 0 && (bestHit.hitDistance < 0 || hit.hitDistance < bestHit.hitDistance)) {
				bestHit = hit;
			}
			}
		if (dot(ray.direction, bestHit.hitNormal) > 0) {
			bestHit.hitNormal = bestHit.hitNormal * -1.0f;
		}
		return bestHit;

	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable * obj : objects) {
			if (obj->intersect(ray).hitDistance > 0) {
				return true;
			}
		}
		return false;
	}

	vec3 Trace(Ray ray, int depth = 0) {
		vec3 outRadiance;
		Hit hit = firstIntersect(ray);
		if(hit.hitDistance<0){
			return Lambient;
		}

		if (!hit.material->isReflective) {  //rough surface
		 outRadiance = hit.material->ambient * Lambient;

			for (Light * light : lights) {
				Ray shadowRay(hit.hitPoint + hit.hitNormal * epsilon, light->direction);
				float cosTheta = dot(hit.hitNormal, light->direction);

				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance = outRadiance + light->energy * hit.material->diffuse * cosTheta;
					vec3 Halfway = normalize(-ray.direction + light->direction);
					float cosDelta = dot(hit.hitNormal, Halfway);

					if (cosDelta > 0) outRadiance = outRadiance + light->energy * hit.material->specular * pow(cosDelta, hit.material->shininess);
				}
			}
		}

		if (hit.material->isReflective) {   //reflective surface
			vec3 relfDirection = normalize(hit.material->reflect(ray.direction, hit.hitNormal));
			Ray reflectRay(hit.hitPoint + normalize(hit.hitNormal) * epsilon, relfDirection);
			outRadiance = outRadiance + Trace(reflectRay, depth + 1)*hit.material->Fresnel(normalize(ray.direction), normalize(hit.hitNormal));
		}

		return outRadiance;
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = Trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}
};


class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture * pTexture;
public:
	void Create(std::vector<vec4>& image) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1, 1, -1, -1, 1,
			1, -1, 1, 1, -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		pTexture = new Texture(windowWidth, windowHeight, image);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		pTexture->SetUniform(gpuProgram.getId(), "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 6);	// draw two triangles forming a quad
	}
};




//unsigned int vao;	   // virtual world on the GPU
FullScreenTexturedQuad fullScreenTexturedQuad;
Scene scene;


//Material(vec3 ka, vec3 ks, vec3 kd, vec3 _n, vec3 _k, float shine, bool ir)
Material GoldMaterial(vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9), 0.0f, true);
Material SilverMaterial(vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0.14, 0.16, 0.13), vec3(4.1, 2.3, 3.1), 0.0f, true);
Material ObjMaterial(vec3(0.4, 0.6, 0.8), vec3(1.0, 0.5, 0.5), vec3(0.05, 0.3, 0.8), vec3(0, 0, 0), vec3(0, 0, 0), 10.0f, false);
Material ObjMaterial2(vec3(0.8, 0.6, 0.4), vec3(1.0, 0.5, 0.5), vec3(0.1, 0.8, 0.3), vec3(0, 0, 0), vec3(0, 0, 0), 10.0f, false);
Material WhiteMaterial(vec3(1, 1, 1), vec3(0, 0, 0), vec3(1, 1, 1), vec3(0, 0, 0), vec3(0, 0, 0), 0.0f, false);

std::vector<Plane> mirrors;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	
	

	//Sphere(const vec3& _center, float _radius, Material* _material)
	Sphere regularS(vec3(0.0f, 0.0f, 0.0f), 0.04f, &ObjMaterial );
	Sphere regularS2(vec3(0.1f, -0.05f, -0.02f), 0.03f, &ObjMaterial2);
	//Sphere mirrorSphere(vec3(0.4f, -0.2f, 0.0f), 0.3f, &SilverMaterial);
	//Sphere mirrorSphere2(vec3(-0.4f, -0.2f, 0.0f), 0.3f, &GoldMaterial);

	vec3 lightDirection(0, 0, 2), Le(1, 1, 1);
	Light light(lightDirection, Le);

	//Plane(const vec3& _point, const vec3& _normal, Material* mat)
	Plane plane(vec3(0.0, 0.3, 0.0), vec3(-0.259, -0.15, 0.0), &GoldMaterial);
	Plane plane2(vec3(0.0, 0.3, 0.0), vec3(0.259, -0.15, 0.0), &GoldMaterial);
	Plane plane3(vec3(0.0, -0.15, 0.0), vec3(0.0, 1.0 , 0.0), &GoldMaterial);
	//Plane bottom(vec3(-0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), &WhiteMaterial);


	

	scene.addObject(&regularS);
	scene.addObject(&regularS2);
	//scene.addObject(&mirrorSphere);
	//scene.addObject(&mirrorSphere2);
	////scene.addObject(&plane);
	//mirrors.push_back(plane);
	//scene.addObject(&plane2);
	//mirrors.push_back(plane2);
	//scene.addObject(&plane3);
	scene.addLight(&light);

	

	scene.build();
	std::vector<vec4> image(windowWidth * windowHeight);

	printf("%d ", scene.objects.size());

	scene.render(image);

	fullScreenTexturedQuad.Create(image);

	

	/*
	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
	float vertices[] = { -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f };
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vertices),  // # bytes
		vertices,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	// create program for the GPU
	
	*/
	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");

}

// Window has become invalid: Redraw
void onDisplay() {

	/*
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
		                      0, 1, 0, 0,    // row-major!
		                      0, 0, 1, 0,
		                      0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
	*/
	//glBindVertexArray(vao);  // Draw call
	//glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 /*# Elements*/);

	fullScreenTexturedQuad.Draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	//if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {


}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system

}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	//long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
