#pragma  once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

#define SUBPROBLEMSIZE 100
/************************************************************************/
/* ���size��ָcol������row          
/* leaf_left  true left ,false right
/************************************************************************/
bool device_diagonal();

#ifndef   GPUSave
class GPUSave 
{
public:
	GPUSave * left;
	GPUSave *right;
	GPUSave *father;
	double *diagonal;
	double *offdiagonal;
	double ak;
	double bk;
	
	int size;
	int deep;
	bool leaf_left;

	double *  left_Q;
	double *  left_q;
	double *  left_D;
	double *  left_W;
	int left_size;

	double * right_Q;
	double * right_q;
	double * right_D;
	double * right_W;
	int right_size;


	double * self_Q;
	double * self_q;
	double * self_D;
	double * self_W;

	GPUSave();
	bool set_left_leaf(double *temp_left_Q,double *temp_left_q,double *temp_left_D,double *temp_left_W,int left_size);
	bool set_right_leaf(double *temp_right_Q,double *temp_right_q,double *temp_right_D,double *temp_right_W,int right_size);

};

 /************************************************************************/
 /* devide_metrix������һ���ܵĺ������Ѵ�ԽǴ����������Ȱ�������� devide_metrix_tree ���ֽ��һ����
 /* Ȼ������ĺ���devide_metrix_tree������˫�ԽǷֽ⣬��������ײ㣬����Ĳ��¼��������Ҫ��¼���ڵ�
 /* ���ڵ�������Ǽ����svd�����ϴ���
 /* Ȼ��read_metrix_tree���������������������ײ�ı�����final_line���cpu����
 /* ��ÿ��Ľڵ㶼��¼��each_deep������ϴ����Ժ��ٽ��м���	
 /* size ����diagonal�Ĵ�С������ Q=(size)*(size+1) q = size+1 D = size*size W = size*size
 /************************************************************************/
template < class T >
void ClearVector( vector< T >& vt );
bool devide_svd_mid(GPUSave *root,int size,double *diagonal, double *offdiagonal );
bool devide_metrix_tree(int size , double * diagonal, double * offdiagonal , int deep, GPUSave * root  );
bool devide_metrix(double * diagonal, double * offdiagonal , int size,vector<vector<GPUSave*> >&each_deep_metrix,vector<GPUSave*>&final_deep_metrix);
bool read_metrix_tree(vector<vector<GPUSave *>> &each_deep, vector<GPUSave *> &final_line, GPUSave * root );
bool save_leaf_metrix(GPUSave *root, int size, int deep , double * diagonal , double * offdiagonal );
bool main_svd(double *diagonal, double *offdiagonal,int size,bool parallel);
bool solve_tree(vector<vector<GPUSave*>>&each_metrix);
void cheack(GPUSave * metrix);
bool solve_tree_parallel(vector<vector<GPUSave*>>&tree);
#endif