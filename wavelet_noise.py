From paper:
@article{10.1145/1073204.1073264,
author = {Cook, Robert L. and DeRose, Tony},
title = {Wavelet Noise},
year = {2005},
issue_date = {July 2005},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {24},
number = {3},
issn = {0730-0301},
url = {https://doi.org/10.1145/1073204.1073264},
doi = {10.1145/1073204.1073264},
abstract = {Noise functions are an essential building block for writing procedural shaders in 3D computer graphics. The original noise function introduced by Ken Perlin is still the most popular because it is simple and fast, and many spectacular images have been made with it. Nevertheless, it is prone to problems with aliasing and detail loss. In this paper we analyze these problems and show that they are particularly severe when 3D noise is used to texture a 2D surface. We use the theory of wavelets to create a new class of simple and fast noise functions that avoid these problems.},
journal = {ACM Trans. Graph.},
month = jul,
pages = {803–811},
numpages = {9},
keywords = {rendering, noise, procedural textures, shading, texture synthesis, texturing, wavelets, multiresolution analysis}
}

https://dl.acm.org/doi/pdf/10.1145/1073204.1073264

/* Note: this code is designed for brevity, not efficiency; many operations can be hoisted,
* precomputed, or vectorized. Some of the straightforward details, such as tile meshing,
* decorrelating bands and fading out the last band, are omitted in the interest of space.*/
static float *noiseTileData; static int noiseTileSize;
int Mod(int x, int n) {int m=x%n; return (m<0) ? m+n : m;}
#define ARAD 16
void Downsample (float *from, float *to, int n, int stride ) {
float *a, aCoeffs[2*ARAD] = {
0.000334,-0.001528, 0.000410, 0.003545,-0.000938,-0.008233, 0.002172, 0.019120,
-0.005040,-0.044412, 0.011655, 0.103311,-0.025936,-0.243780, 0.033979, 0.655340,
0.655340, 0.033979,-0.243780,-0.025936, 0.103311, 0.011655,-0.044412,-0.005040,
0.019120, 0.002172,-0.008233,-0.000938, 0.003546, 0.000410,-0.001528, 0.000334};
a = &aCoeffs[ARAD];
for (int i=0; i<n/2; i++) {
to[i*stride] = 0;
for (int k=2*i-ARAD; k<=2*i+ARAD; k++)
to[i*stride] += a[k-2*i] * from[Mod(k,n)*stride];
}
}
void Upsample( float *from, float *to, int n, int stride) {
float *p, pCoeffs[4] = { 0.25, 0.75, 0.75, 0.25 };
p = &pCoeffs[2];
for (int i=0; i<n; i++) {
to[i*stride] = 0;
for (int k=i/2; k<=i/2+1; k++)
to[i*stride] += p[i-2*k] * from[Mod(k,n/2)*stride];
}
}
void GenerateNoiseTile( int n, int olap) {
if (n%2) n++; /* tile size must be even */
int ix, iy, iz, i, sz=n*n*n*sizeof(float);
float *temp1=(float *)malloc(sz),*temp2=(float *)malloc(sz),*noise=(float *)malloc(sz);
/* Step 1. Fill the tile with random numbers in the range -1 to 1. */
for (i=0; i<n*n*n; i++) noise[i] = gaussianNoise();
/* Steps 2 and 3. Downsample and upsample the tile */
for (iy=0; iy<n; iy++) for (iz=0; iz<n; iz++) { /* each x row */
i = iy*n + iz*n*n; Downsample( &noise[i], &temp1[i], n, 1 );
Upsample( &temp1[i], &temp2[i], n, 1 );
}
for (ix=0; ix<n; ix++) for (iz=0; iz<n; iz++) { /* each y row */
i = ix + iz*n*n; Downsample( &temp2[i], &temp1[i], n, n );
Upsample( &temp1[i], &temp2[i], n, n );
}
for (ix=0; ix<n; ix++) for (iy=0; iy<n; iy++) { /* each z row */
i = ix + iy*n; Downsample( &temp2[i], &temp1[i], n, n*n );
Upsample( &temp1[i], &temp2[i], n, n*n );
}
/* Step 4. Subtract out the coarse-scale contribution */
for (i=0; i<n*n*n; i++) {noise[i]-=temp2[i];}
/* Avoid even/odd variance difference by adding odd-offset version of noise to itself.*/
int offset=n/2; if (offset%2==0) offset++;
for (i=0,ix=0; ix<n; ix++) for (iy=0; iy<n; iy++) for (iz=0; iz<n; iz++)
temp1[i++] = noise[ Mod(ix+offset,n) + Mod(iy+offset,n)*n + Mod(iz+offset,n)*n*n ];
for (i=0; i<n*n*n; i++) {noise[i]+=temp1[i];}
noiseTileData=noise; noiseTileSize=n; free(temp1); free(temp2);
}
