#include <linalg.h>
#include <stdlib.h>
#include <stdio.h>



//�˴��ȴ��Ż�����Ϊ�����ȰѶԽǾ���ת���ɴ�������0����ɣ�Ȼ��������ֵ�ֽ⣬����Ǻܲ��õģ��м򻯵����
namespace gpu_svd{

        /***********************************
		 * ��һ������ֵ�ֽ⣬�ֽ���С�ľ�����cpu��alglib��
		 * ������Ҫ�������󣬰�˫�Խ�����һ�������ľ���
		 * �ٵ��ÿ⺯�����
		 * diagonal �� offdiagonal �ֱ���size-1 ��С�� 1ά���飬��˫�Խǵ�Ԫ��
		 * u,w,vt �Ƿ��ص�����ֵ�ֽ��ľ���
		 * first_svd()�����ֽ���󣬸�˫�Խǵ�ֵ�����д�С������ȥQ��q,D,W 
		 * get_2d_fromalg()�ǰ�real_2d_array��Ķ���������
		 * get_1d_fromalg()�ǰ�real_1d_array��Ķ���������
		 * �����Q,q,D,W���Զ���������ռ䣬������ǰ����ռ�
		 ***********************************/

		 bool first_svd(double *diagonal,double *offdiagonal,int size,double *Q,double *q,double *D,double  *W);
		 bool get_2d_fromalg( int start_row,int row_num,int start_col,int col_num,double *metrix,alglib::real_2d_array *a);
		 bool get_1d_fromalg(int col,double *metrix,alglib::real_1d_array *a);
		 bool increase(int size,double *,double *,double *,double *);
		 bool transpose(int rowNum,int colNum,double *metrix,alglib::real_2d_array *a);
		 bool reMulti(FILE *ph,int size,double *Q ,double *D, double *W);
}	
