#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16


#define volatile //Po usunieciu tej linii program wykonuje sie poprawnie(pare zmiennych znika bez powodu?)


void __global__ gmandelbrot(int* Mandel,
		int poz,
		int pion,
		int max,
		float x0,
		float y0,
		float x1,
		float y1) {
	int JUMP =2;
	float dx = (x1-x0)/(poz-1);
	float dy = (y1-y0)/(pion-1);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float xf = x0+x*dx;
	float yf = y0+y*dy;
	int i = 0;
	int li = 0;
	float xz = xf;
	float yz = yf;
	volatile float xzs = xf*xf;
	volatile float yzs = yf*yf;
	volatile float lxz = xf;
	float lyz = yf;
	while((i<max) && ((xzs + yzs) < 4.0))
	{

		lyz = yz;
		lxz = xz;
		li = i;
		for(int j = 0; j < JUMP; j++) {
			yz = xz*yz*2 + yf;
			xz = xzs-yzs+xf;
			xzs = xz*xz;
			yzs = yz*yz;
		}
		i=i  +JUMP;
		JUMP = JUMP + 8;
	}
	xz=  lxz;
	yz = lyz;
	xzs = xz * xz;
	yzs = yz*yz;
	i = li;
	while((i<max) && ((xzs+yzs) < 4.0)) 
	{
		yz = xz*yz*2 + yf;
		xz = xzs-yzs+xf;
		xzs = xz*xz;
		yzs = yz*yz;
		i++;
	}
	Mandel[(pion-y-1)*poz + x] = i;
}



void  mandelbrot1(int* Mandel, int width, int height, int max, float x0, float y0, float x1, float y1) {
	dim3 grid(width / BLOCK_WIDTH, height / BLOCK_HEIGHT, 1);
	dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
	gmandelbrot<<<grid, block>>>(Mandel, width, height, max, x0, y0, x1, y1);
}

#endif
