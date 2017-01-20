#include "devide_metrix.h"
#include <queue>
#include "caculate_metrix.h"
#include "Myvector.h"
#include "first_svd.h"
#include "print_metrix.h"
#include "secular_equation.cuh"
#include <time.h>
#include <vector>
bool devide_metrix_tree(int size , double * diagonal, double * offdiagonal , int deep, GPUSave * root  )
{
	if(size<=0)
	{
		printf("it's error in devide_metrix and the size <=0");
		return false;
	}
	if(size<=SUBPROBLEMSIZE)  //ֻ����С��SUBPROLEMSIZE��ֹͣ�ݹ�
	{
		save_leaf_metrix(root,size,deep,diagonal,offdiagonal);
		return true;
	}
	else
	{
		int size_left = size/2;
		int size_right = size-size_left-1;
		GPUSave *temp_left ;
		temp_left= new GPUSave;
		GPUSave *temp_right; 
		temp_right= new GPUSave;
	
		temp_right->father = root;              //true and false �����������ҵ�
		temp_right->leaf_left = false;
		temp_left->father = root;
		temp_left->leaf_left = true;
		devide_metrix_tree(size_left,diagonal,offdiagonal,deep+1,temp_left);
		devide_metrix_tree(size_right,diagonal+size_left+1,offdiagonal+size_left+1,deep+1,temp_right);//����ط������޸ģ������϶������󣬲����ٸĻ���
		root->ak = *(diagonal+size_left);   
		root->bk = *(offdiagonal+size_left);
		root->deep = deep;
		root->left =temp_left;
		root->right = temp_right;
		return true;
	}
}
bool save_leaf_metrix(GPUSave *root, int size, int deep , double * diagonal , double * offdiagonal )
{
	if(size<=0)
	{
		printf("the save_leaf_metrix is error\n");
		return 0;
	}
	root->deep = deep;
	root->diagonal = diagonal;
	root->offdiagonal = offdiagonal ; 
	root->size = size;
	
	root->left = NULL;
	root->right = NULL;

	return true;
}




bool devide_metrix(double * diagonal, double * offdiagonal , int size,vector<vector<GPUSave * > > &each_deep_metrix , vector<GPUSave*> &final_deep_metrix)
{	

	GPUSave *root;
	if(size<=0)
	{
		return false;
	}
	if(size <SUBPROBLEMSIZE)
	{
		root =new GPUSave;
		double * dev_Q =(double*)malloc(sizeof(double)*size*(size+1));
		double * dev_q =(double*)malloc(sizeof(double)*(size+1));
		double * dev_D =(double*)malloc(sizeof(double)*size*size);
		double * dev_W=(double*)malloc(sizeof(double)*size*size);
		gpu_svd::first_svd(diagonal,offdiagonal,size,dev_Q,dev_q,dev_D,dev_W);
		root->self_Q = dev_Q;
		root->self_D = dev_D;
		root->self_q = dev_q;
		root->self_W= dev_W;
		final_deep_metrix.push_back(root);
		return true;
	}
	root = new GPUSave;
	root->father = NULL;
	
	devide_metrix_tree(size,diagonal,offdiagonal,1,root);
	read_metrix_tree(each_deep_metrix,final_deep_metrix,root);
	return true;
}
/*deep �Ǵ�1��ʼ��*/
bool read_metrix_tree(vector<vector<GPUSave *> > &each_deep, vector<GPUSave *> &final_line, GPUSave * root)
{
	queue<GPUSave *> temp_queue;
	temp_queue.push(root);
	while (!temp_queue.empty())
	{
		GPUSave * temp = temp_queue.front();
		temp_queue.pop();
		if (temp->left==NULL&&temp->right==NULL)
		{
			final_line.push_back(temp);
		} 
		else if(temp->left!=NULL&&temp->right!=NULL)
		{
			while(each_deep.size() < (unsigned int)temp->deep) //���if��Ϊwhile��
			{
				vector<GPUSave *> tempmid;
				each_deep.push_back(tempmid);
							
			}
			each_deep[temp->deep-1].push_back(temp);
			temp_queue.push(temp->left);
			temp_queue.push(temp->right);
		}
		else
		{
			if(temp->left ==NULL)
			{
				printf("Error : the left leaf is null in devide_metrix!\n");
			}
			if(temp->right==NULL)
			{
				printf("Error : the right leaf is null in devide_metrix\n");

			}
			return false;
		}

	}
	return true;

}


GPUSave::GPUSave()
{
	size = 0;
	ak = 0;
	bk = 0;
	deep = 0;
	left = NULL;
	right = NULL;
	left_Q = NULL;
	left_q = NULL;
	left_D = NULL;
	left_W = NULL;
	right_Q = NULL;
	right_q = NULL;
	right_D = NULL;
	right_W = NULL;
	self_W = NULL;
	self_Q = NULL;
	self_D = NULL;
	self_q = NULL;
	diagonal = NULL;
	offdiagonal = NULL;
	father = NULL;
	left_size = 0;
	right_size = 0;
	leaf_left = false;
}

bool GPUSave::set_left_leaf(double *temp_left_Q,double *temp_left_q,double *temp_left_D,double *temp_left_W,int size)
{
	if ( temp_left_Q==NULL|| temp_left_q==NULL||temp_left_D==NULL||temp_left_W==NULL)
	{
		return false;
	}
	left_Q = temp_left_Q;
	left_q = temp_left_q;
	left_D = temp_left_D;
	left_W = temp_left_W;
	left_size = size;
	return true;	
}

bool GPUSave::set_right_leaf(double *temp_right_Q,double *temp_right_q,double *temp_right_D,double *temp_right_W,int size)
{
	if ( temp_right_Q==NULL|| temp_right_q==NULL||temp_right_D==NULL||temp_right_W==NULL)
	{
		return false;
	}
	right_Q = temp_right_Q;
	right_q = temp_right_q;
	right_D = temp_right_D;
	right_W =temp_right_W;
	right_size = size;
	return true;
}
/*size is the diagonal'size*/
bool devide_svd_mid(GPUSave *root,int size,double *diagonal,double *offdiagonal)
{
	double * dev_Q;
	double * dev_q;
	double * dev_D;
	double * dev_W;
	
	dev_Q = (double *)malloc(sizeof(double)*size*(size+1));
	dev_q = (double *)malloc(sizeof(double)*(size+1));
	dev_W = (double *)malloc(sizeof(double)*size*size);
	dev_D = (double *)malloc(sizeof(double)*size*size);
	
	gpu_svd::first_svd(diagonal,offdiagonal,size,dev_Q,dev_q,dev_D,dev_W);
	if(root->father==NULL)
	{
		return false;
	}
	GPUSave * father = root->father;
	if(root->leaf_left==true)
	{
		father->set_left_leaf(dev_Q,dev_q,dev_D,dev_W,size);
	}
	else
	{
		father->set_right_leaf(dev_Q,dev_q,dev_D,dev_W,size);
	}
	return true;
}

bool main_svd(double * diagonal, double *offdiagonal,int size,bool parallel)
{
	if(size<SUBPROBLEMSIZE)
	{
		/************************************************
		 * ������С���󣬵���ʼ����С�ڸ�ֵ����cpuֱ�Ӽ���
		 *
		 *************************************************/
		double *dev_Q;
		double *dev_q;
		double *dev_D;	
		double *dev_W;
		dev_Q = (double *)malloc(sizeof(double)*size*(size+1));
		dev_q = (double *)malloc(sizeof(double)*(size+1));
		dev_W = (double *)malloc(sizeof(double)*size*size);
		dev_D = (double *)malloc(sizeof(double)*size*size);
		gpu_svd::first_svd(diagonal,offdiagonal,size,dev_Q,dev_q,dev_D,dev_W);
		free(dev_Q);
		free(dev_q);
		free(dev_W);
		free(dev_D);
		return true;
	}
	vector <vector<GPUSave *> >each_deep_metrix; //����������һ������в�����
	vector<GPUSave*>final_deep_metrix;  //�������һ�������
	devide_metrix(diagonal,offdiagonal,size,each_deep_metrix,final_deep_metrix);
	clock_t start=clock();
	clock_t end;
	clock_t cputime;
	caculate_final_svd(final_deep_metrix);
	cputime=clock();
	printf("the first svd takes %lfs\n",(cputime-start)*1.0/CLOCKS_PER_SEC);
	if(!parallel)
	{
		solve_tree(each_deep_metrix);
		end=clock();
		printf("��cpu+��gpu���㻨��ʱ��Ϊ%lf\n",(end-start)*1.0/CLOCKS_PER_SEC);
	}
	else
	{
		solve_tree_parallel(each_deep_metrix);
		end=clock();
		printf("��cpu+˫gpu���㻨��ʱ��Ϊ%lf\n",(end-start)*1.0/CLOCKS_PER_SEC);
	}
//	cheack(each_deep_metrix[0][0]);
	ClearVector(final_deep_metrix);
	return true;
}

template < class T >
void ClearVector( vector< T >& vt )
{
	vector< T > vtTemp;
	vtTemp.swap( vt );
}
//��û������洢vector
bool solve_tree(vector<vector<GPUSave*>>&each_metrix)
{

	int deep_all=each_metrix.size();
	for(int i=deep_all-1;i>0;i--)
	{
		int each_size=each_metrix[i].size();
		for(int j=0;j<each_size;++j)
		{
			produce_mid_metrix(each_metrix[i][j],1,false);
		}
		printf("�������%d��\n",i+1);

	}
	produce_mid_metrix(each_metrix[0][0],1,true);
	
	return true;

}
bool solve_tree_parallel(vector<vector<GPUSave*>>&tree)
{
	int deep=tree.size();
	for(int i=deep-1;i>0;i--)
	{
		int size=tree[i].size();
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				for(int j=0;j<size/2;++j)
				{
					produce_mid_metrix(tree[i][j],0,false);
				}
			}
			#pragma omp section
			{
				for(int j=size/2;j<size;++j)
				{
					produce_mid_metrix(tree[i][j],1,false);
				}
			}
		}
		printf("�������%d��\n",i+1);
	}
	produce_mid_metrix(tree[0][0],0,true);
	return 1;
}
void cheack(GPUSave * metrix)
{
	FILE *ph;
	fopen_s(&ph,"F:/result.txt","w");
	clock_t times=clock();
	reCheack(ph,metrix->self_Q,metrix->self_q,metrix->self_D,metrix->self_W,metrix->size);
	printf("the cpu solve the size=%d problem takes %lfs\n",metrix->size,(clock()-times)*1.0/CLOCKS_PER_SEC);
	fclose(ph);	
}










