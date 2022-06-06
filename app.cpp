#include "app.h"

float anim_time = 0.0f;
float anim_dt = 0.03f;

unsigned int window_width;
unsigned int window_height;

int mouse_old_x, mouse_old_y;
bool motion_p;

double phi = -90.0;
double theta = 90.0;

double eye[3];
double center[3] = {0.0, 0.0, 0.0};
double up[3];

GLuint vao, vbo_pos, vbo_index, vbo_norm;
unsigned short point_index[NUM_INDEX];
unsigned short *d_index;

float h_point[NUM_POINTS][3];
float (*d_point)[3];
static float3 *point = NULL;

float h_pointPre[NUM_POINTS][3];
float (*d_pointPre)[3];
static float3 *pre = NULL;

float h_weight[NUM_POINTS];
float *d_weight;
static float3 *weight = NULL;

float h_rest[DIV_H][6];
float (*d_rest)[6];
static float3 *rest = NULL;

float h_norm_surf[NUM_INDEX - 2][3];
float (*d_norm_surf)[3];

float h_norm[NUM_POINTS][3];
float (*d_norm)[3];

struct cudaGraphicsResource *vbo_res, *vbo_norm_res;

extern void launchGPUKernel(float (*point)[3], float (*pre)[3], float *weight, float (*rest)[6], float time, float dt, float (*d_norm)[3], float (*d_norm_surf)[3], unsigned short *point_index);

void app::setup(void)
{
	//openGLの設定
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glPointSize(4.0f);

	//ライトの設定
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	float light_pos[4] = {0.3f, 1.0f, 0.0f, 1.0f};
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
	float light_amb[4] = {0.2f, 0.2f, 0.2f, 1.0f};
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_amb);
	float light_dif[4] = {0.2f, 0.4f, 1.0f, 1.0f};
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_dif);
	float light_spe[4] = {0.8f, 0.8f, 1.0f, 1.0f};
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_spe);

	//マテリアルの設定
	float material_amb[4] = {0.2f, 0.2f, 0.2f, 1.0f};
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_amb);
	float material_dif[4] = {0.8f, 0.8f, 0.8f, 1.0f};
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_dif);
	float material_spe[4] = {0.5f, 0.5f, 0.5f, 1.0f};
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_spe);
	float material_emi[4] = {0.0f, 0.0f, 0.0f, 1.0f};
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, material_emi);
	glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 5);

	//GPUメモリを確保
	cudaMalloc((void**)&d_pointPre, sizeof(float) * NUM_POINTS * 3);
	cudaMalloc((void**)&d_weight, sizeof(float) * NUM_POINTS);
	cudaMalloc((void**)&d_rest, sizeof(float) * DIV_H * 6);
	cudaMalloc((void**)&d_index, sizeof(unsigned short) * NUM_INDEX);
	cudaMalloc((void**)&d_norm_surf, sizeof(float) * (NUM_INDEX - 2) * 3);

	//頂点初期化
	int p_i, s_i, i_i, offset;
	i_i = 0;
	for (unsigned int j = 0; j < DIV_H; j++){
		for (unsigned int i = 0; i < DIV_W; i++){
			p_i = j * DIV_W + i;

			//初期位置設定
			h_point[p_i][X] = h_pointPre[p_i][X] = (sqrt(0.0625f - (0.5f * (float)j / (float)DIV_H - 0.25f) * (0.5f * (float)j / (float)DIV_H - 0.25f)) + (0.25f - (-0.5f * (float)j / (float)DIV_H + 0.25f)) / 2.0f) * cos((float)i * 2.0f * PI / (float)DIV_W);
			h_point[p_i][Y] = h_pointPre[p_i][Y] = -0.5f * (float)j / (float)DIV_H + 0.25f;
			h_point[p_i][Z] = h_pointPre[p_i][Z] = (sqrt(0.0625f - (0.5f * (float)j / (float)DIV_H - 0.25f) * (0.5f * (float)j / (float)DIV_H - 0.25f)) + (0.25f - (-0.5f * (float)j / (float)DIV_H + 0.25f)) / 2.0f) * sin((float)i * 2.0f * PI / (float)DIV_W);

			//固定点を設定
			if (j == 0)
				h_weight[p_i] = 0.0f;
			else
				h_weight[p_i] = 1.0f;

			//頂点インデックスを設定
			if(j != DIV_H - 1){
				point_index[i_i] = j * DIV_W + i;
				i_i++;
				point_index[i_i] = (j + 1) * DIV_W + i;
				i_i++;
				if(i == DIV_W - 1){
					point_index[i_i] = j * DIV_W;
					i_i++;
					point_index[i_i] = (j + 1) * DIV_W;
					i_i++;
				}
			}
		}

		//バネの自然長計算
		s_i = j * DIV_W;
		offset = j * DIV_W + 1;
		h_rest[j][0] = distance(h_point[s_i], h_point[offset]);
		offset = j * DIV_W + 2;
		h_rest[j][3] = distance(h_point[s_i], h_point[offset]);
		offset = j * DIV_W + DIV_W / 2;
		h_rest[j][5] = distance(h_point[s_i], h_point[offset]);
		if(j != 0){
			offset = (j - 1) * DIV_W;
			h_rest[j][1] = distance(h_point[s_i], h_point[offset]);
			offset = (j - 1) * DIV_W + 1;
			h_rest[j][2] = distance(h_point[s_i], h_point[offset]);
			if(j != 1){
				offset = (j - 2) * DIV_W;
				h_rest[j][4] = distance(h_point[s_i], h_point[offset]);
			}
		}
	}

	//法線計算
	for(unsigned int i = 0; i < NUM_INDEX - 2; i += 2){
		normal(h_point[point_index[i]], h_point[point_index[i + 2]], h_point[point_index[i + 1]], h_norm_surf[i]);
		normal(h_point[point_index[i + 1]], h_point[point_index[i + 2]], h_point[point_index[i + 3]], h_norm_surf[i + 1]);
	}

	//vao生成
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	//頂点vbo生成
	glGenBuffers(1, &vbo_pos);
 	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
 	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * NUM_POINTS * 3, h_point, GL_DYNAMIC_DRAW);
	glEnableVertexArrayAttrib(vao, 0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
 	cudaGraphicsGLRegisterBuffer(&vbo_res, vbo_pos, cudaGraphicsRegisterFlagsNone);

	//インデックスvbo生成
	glGenBuffers(1, &vbo_index);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_index);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * NUM_INDEX, &point_index, GL_STATIC_DRAW);

	//法線vbo生成
	glGenBuffers(1, &vbo_norm);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * NUM_POINTS * 3, h_norm, GL_DYNAMIC_DRAW);
	glEnableVertexArrayAttrib(vao, 2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
	cudaGraphicsGLRegisterBuffer(&vbo_norm_res, vbo_norm, cudaGraphicsRegisterFlagsNone);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	//GPUメモリにコピー
	cudaMemcpy(d_pointPre, h_pointPre, sizeof(float) * NUM_POINTS * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, h_weight, sizeof(float) * NUM_POINTS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rest, h_rest, sizeof(float) * DIV_H * 6, cudaMemcpyHostToDevice);
	cudaMemcpy(d_index, point_index, sizeof(unsigned short) * NUM_INDEX, cudaMemcpyHostToDevice);
	cudaMemcpy(d_norm_surf, h_norm_surf, sizeof(float) * (NUM_INDEX - 2) * 3, cudaMemcpyHostToDevice);
}

void app::update(void)
{
	cudaGraphicsMapResources(1, &vbo_res, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_point, NULL, vbo_res);
	cudaGraphicsMapResources(1, &vbo_norm_res, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_norm, NULL, vbo_norm_res);
	launchGPUKernel(d_point, d_pointPre, d_weight, d_rest, anim_time, anim_dt, d_norm, d_norm_surf, d_index);

	cudaGraphicsMapResources(1, &vbo_res, 0);
	cudaGraphicsUnmapResources(1, &vbo_norm_res, 0);

	anim_time += anim_dt;
	glutPostRedisplay();
}

void app::draw(void)
{
	defineViewMatrix(phi, theta);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindVertexArray(vao);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	//クラゲの頭
	glEnable(GL_LIGHTING);
	glDrawElements(GL_TRIANGLE_STRIP, NUM_INDEX, GL_UNSIGNED_SHORT, 0);
	glDisable(GL_LIGHTING);

	glColor3f(0.8f, 0.9f, 1.0f);
	glDrawArrays(GL_POINTS, 0, NUM_POINTS);

	glBindVertexArray(0);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

}

// 視点座標計算
void app::defineViewMatrix(double phi, double theta)
{
        unsigned int i, j;
        double c, s, xy_dist;

        eye[Z] = sin(theta * PI / 180.0);
        xy_dist = cos(theta * PI / 180.0);
        c = cos(phi * PI / 180.0);
        s = sin(phi * PI / 180.0);
        eye[X] = xy_dist * c;
        eye[Y] = xy_dist * s;
        up[X] = -c * eye[Z];
        up[Y] = -s * eye[Z];
        up[Z] = s * eye[Y] + c * eye[X];

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60.0, (double)window_width / window_height, 0.1, 100.0);
        glViewport(0, 0, window_width, window_height);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(eye[X], eye[Y], eye[Z], center[X], center[Y], center[Z], up[X], up[Y], up[Z]);
}


void app::keyPressed(unsigned char key)
{

}

void app::mousePressed(int x, int y, int button, int state)
{
	if((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON))
		motion_p = true;
	else if(state == GLUT_UP)
		motion_p = false;
	mouse_old_x = x;
	mouse_old_y = y;
}

void app::mouseDragged(int x, int y)
{
	int dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;
	if(motion_p){
		phi -= dx * 0.2;
		theta += dy * 0.2;
	}
	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void app::windowResize(int width, int height)
{
	window_width = width;
	window_height = height;
}

void app::end(void)
{
	cudaGraphicsUnregisterResource(vbo_res);
	glDeleteBuffers(1, &vbo_pos);
	glDeleteBuffers(1, &vbo_index);
	glDeleteBuffers(1, &vbo_norm);
	glDeleteVertexArrays(1, &vao);
	vbo_pos = 0;
	vbo_index = 0;
	vbo_norm = 0;
	vao = 0;
	cudaFree(d_point);
	cudaFree(d_pointPre);
	cudaFree(d_weight);
	cudaFree(d_rest);
	cudaFree(d_norm);
	cudaDeviceReset();
}
