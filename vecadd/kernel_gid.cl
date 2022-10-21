__kernel void vecadd (__global const float *A,
	                    __global const float *B,
	                    __global float *C)
{
  int gid = get_global_id(0);
  if(A[gid] <= 0.5){
    C[gid] = A[gid] + B[gid];
  }
  else {
    C[gid] = 0;
  }
}
