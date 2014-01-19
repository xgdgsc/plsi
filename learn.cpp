#include<boost/python.hpp>
#include<iostream>
#include<cmath>
#include<vector>
#include <cassert>

using namespace boost::python;
using namespace std;


double loglikelihood(list featureList,int docNum,int wordNum,int z,list p_wz_n,list p_dz_n,list p_z_n)
{
  //cout<<"z:"<<z<<endl;
//  cout<<"docNum:"<<docNum<<endl;
//  cout<<"wordNum:"<<wordNum<<endl;

  double new_log_like=0;
  for(size_t d=0;d<docNum;d++)
    {
     // std::cout<<featureList[d]<<std::endl;
      dict doc=extract<dict>(featureList[d]);
      list ks=doc.keys();
      size_t docWord=len(ks);
//      cout<<"docWord:"<<docWord<<endl;
//      cout<<"d:"<<d<<' ';
      for(size_t j=0;j<docWord;j++)
        {
          int w=extract<int>(ks[j]);
//          cout<<"w:"<<w<<':';
          int tfwd=extract<int>(doc[w]);
//          cout<<tfwd<<' ';
          w=w-1;
          double p_d_w=0;
          //cout<<"p:"<<p_wz_n[w][2]<<endl;
          for(size_t i=0;i<z;i++)
            {
              //cout<<"pwzn: "<<extract<double>(p_wz_n[w][i])<<endl;
              p_d_w+=extract<double>(p_wz_n[w][i])*extract<double>(p_dz_n[d][i])*extract<double>(p_z_n[i]);
              //cout<<p_d_w<<endl;
            }
          new_log_like+=tfwd*log(p_d_w);

        }
//      cout<<endl;
    }

  return new_log_like;
     
}

list update(list featureList,int docNum,int wordNum,int z,list p_wz_n,list p_dz_n,list p_z_n)
{
  list res=list();
  vector<vector<double> > numerator_p_dz_n(docNum,vector<double>(z,0));
  vector<vector<double> > numerator_p_wz_n(wordNum,vector<double>(z,0));
  vector<double> numerator_p_z_n=vector<double>(z,0);
  vector<double> denominator_p_dz_n=vector<double>(z,0);
  vector<double> denominator_p_wz_n=vector<double>(z,0);
  double denominator_p_z_n=0;

  for(size_t d=0;d<docNum;d++)
    {
      dict doc=extract<dict>(featureList[d]);
      list ks=doc.keys();
      size_t docWord=len(ks);
      for(size_t j=0;j<docWord;j++)
        {
          int w=extract<int>(ks[j]);
          int tfwd=extract<int>(doc[w]);
          w=w-1;
          double denominator=0;
          double numerator[z];
          for(size_t i=0;i<z;i++){
              numerator[i]=extract<double>(p_dz_n[d][i])*extract<double>(p_wz_n[w][i])*extract<double>(p_z_n[i]);
              assert(numerator[i]>=0);
              denominator+=numerator[i];
            }
          //double P_z_condition_d_w[z];
          for(size_t i=0;i<z;i++){
              double P_z_condition_d_w=numerator[i]/denominator;
              numerator_p_wz_n[w][i]+=tfwd*P_z_condition_d_w;
              denominator_p_wz_n[i]+=tfwd*P_z_condition_d_w;
              numerator_p_dz_n[d][i]+=tfwd*P_z_condition_d_w;
              denominator_p_dz_n[i]+=tfwd*P_z_condition_d_w;
              numerator_p_z_n[i]+=tfwd*P_z_condition_d_w;

            }
          denominator_p_z_n+=tfwd;

        }
    }
    //update p_wz_n
  for(size_t w=0;w<wordNum;w++)
    {
      for(size_t i=0;i<z;i++){
          p_wz_n[w][i]=numerator_p_wz_n[w][i]/denominator_p_wz_n[i];
        }
    }

//update p_dz_n
  for(size_t d=0;d<docNum;d++)
    {
       for(size_t i=0;i<z;i++){
           p_dz_n[d][i]=numerator_p_dz_n[d][i]/denominator_p_dz_n[i];
         }
    }

  //update p_z_n
  for(size_t i=0;i<z;i++)
    {
      p_z_n[i]=numerator_p_z_n[i]/denominator_p_z_n;
    }

  res.append(p_wz_n);
  res.append(p_dz_n);
  res.append(p_z_n);
  return res;
}

BOOST_PYTHON_MODULE(learnC)
{

  def("loglikelihood",loglikelihood);
  def("update",update);
}








