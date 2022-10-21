
__kernel void SobelDetector(__global uchar* input, __global uchar* output) {
	uint x = get_global_id(0);
    	uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);
	int w = width;
	int i =  w*y + x ;
	int gx = 0 ;
	int gy = 0 ;
	int gx_idx[6] = {1,2,1,-1,-2,-1};
	int gy_idx[6] = {1,-1,2,-2,1,-1};
	for (int k=0;k<6;k++)
	{
		if(k<3){
			gx += *(input+i-w-4 + k*4) * gx_idx[k];
		}
		else{
			gx += *(input+i+w-4 + (k-3)*4) * gx_idx[k];
		}
	}
	for (int k=0;k<6;k++)
	{
		if(k<2){
			gy += *(input+i-w-4 + k*8) * gy_idx[k];
		}
		else if(k<4 && k>=2){
			gy += *(input+i-4 + (k-2)*8) * gy_idx[k];
		}
		else{
			gy += *(input+i+w-4 + (k-4)*8) * gy_idx[k];
		}
	}

        int gx2 = gx*gx;
        int gy2 = gy*gy;
	//float gsum = (float)(gx2+gy2) ;
	//int gout = (int)sqrt(gsum);
	//output[i] = (uchar)0;
	//for(int ii=1;ii<3;ii++){
	//	int gout = (int)sqrt(gsum);
        //output[i] = gout / 2 ;
	//}	
        output[i] = (abs(gx)+abs(gy) )/2 ;
}

	

	 






	

	




	

	

	
	
