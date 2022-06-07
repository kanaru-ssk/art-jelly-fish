#include "main.h"
//#include "app.h"

// アイドル状態の処理
void idle(void)
{
	app::update();
}

// 描画処理 
void display(void)
{
	app::draw();
}

// ウィンドウリサイズ時の処理
void resize(int width, int height)
{
	app::windowResize(width, height);
}

// キーボード押下時の処理
void keyboard(unsigned char key, int x, int y)
{
	app::keyPressed(key);
}

// クリック時の処理
void mouse_button(int button, int state, int x, int y)
{
        app::mousePressed(x, y, button, state);
}

// ドラッグ時の処理
void mouse_motion(int x, int y)
{
        app::mouseDragged(x, y);
}

void end(void)
{
        app::end();
}

// 起動時の処理
void init(int *argc, char **argv)
{
	glutInit(argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutInitWindowPosition(INIT_X_POS, INIT_Y_POS);
        glutInitWindowSize(INIT_WIDTH, INIT_HEIGHT);
        glutCreateWindow("Jelly-Fish");
        glutIdleFunc(idle);
        glutDisplayFunc(display);
        glutReshapeFunc(resize);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse_button);
        glutMotionFunc(mouse_motion);
        glutCloseFunc(end);

	glewExperimental = GL_TRUE;
	glewInit();
	
	app::setup();
}

int main(int argc, char **argv)
{
	init(&argc, argv);
	glutMainLoop();
	return 0;
}
