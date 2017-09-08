#include<boost/python.hpp>
#include<boost/python/numeric.hpp>
#include<boost/python/tuple.hpp>
#include<vector>
#include<cmath>
#include "glog/logging.h"
#include<algorithm>

/*
          C++ interface for Python

        they say tree has been widely used
Chen Yliang
Sep 5, 2017
*/
namespace Py = boost::python;
const float postive_zero = 1E-006;
class Node
{
public:
  Node(Py::list &labels):labels(labels),class_num(Py::len(labels)), loss(INFINITY){};
  Node(Py::numeric::array &A, Py::list &L, int class_num);
  Node(const Node &){throw -1;};
//  int cur_level;
//  int max_level;
  int feature_idx;           //  the dimemsion targeted by this node
  int feature_dims;         // length of the vector, the LAST ROW of the sample_array is label!
  int class_num;

  Py::list *sample_list_ptr;      //  indexs for training used in this node, DO NOT use auto_ptr ! since from ref
  Py::numeric::array *sample_array_ptr; // all of the samples

//  Py::numeric::array &  label_array;
//  Py::numeric::array & ;
  float loss;
  Py::list labels;  // if this node is not the leaf then, it has many labels. if not single label
  float threshold;
//  std::vector<float> 
  std::auto_ptr<Node> left_ptr;
  std::auto_ptr<Node> right_ptr;

//  inline Py::list& get_left_labels(){return left_labels;};
//  inline Py::list& get_right_labels(){return right_labels;};

  Py::list left_labels;      //   tags for the left
  Py::list right_labels;

  void train();
  Py::list predict(Py::numeric::array &feature);
  

private:
  void set_tags();              // ONLY after train()!   update {left/right}_labels 
  float calc_loss(float th, int dim_i); // return the loss for the given threshold
  std::auto_ptr<float> label_R_cnt_ptr;   //  pre-allocate for calc_loss
  std::auto_ptr<float> label_L_cnt_ptr;

  std::auto_ptr<float> loss_R_cnt_ptr;  // always stores the count related to the current min loss, for set_tags()
  std::auto_ptr<float> loss_L_cnt_ptr;

};//end of CLASS Tree






Node::Node(Py::numeric::array &A, Py::list &L, int class_num):loss(INFINITY),class_num(class_num)
{
  /*
      since data has been got, train it.
  */
  Py::tuple shape = Py::extract<Py::tuple>(A.attr("shape"));
  feature_dims = Py::extract<int>(shape[0])-1;
  label_R_cnt_ptr.reset( new float[class_num] );
  label_L_cnt_ptr.reset( new float[class_num] );

  loss_R_cnt_ptr.reset(new float[class_num]);
  loss_L_cnt_ptr.reset(new float[class_num]);

  sample_list_ptr = &L;
  sample_array_ptr = &A;

  train();
}

void Node::train()
{
  int list_len = Py::len(*sample_list_ptr);
  if( list_len == 0 || Py::len(*sample_array_ptr) == 0)
    return;
  
  // begin loop
  float th_tmp(0.);
  float loss_tmp(0);
  for(int dim_i =0; dim_i<feature_dims; ++dim_i)
  {
//    LOG(INFO)<<"dim_i: "<<dim_i;
    for(int list_i = 0; list_i< list_len; ++list_i)
    {
      int sample_i = int( Py::extract<float>( (*sample_list_ptr)[list_i])) ;
//      LOG(INFO) << "passing sample_i";
      th_tmp = Py::extract<float>( (*sample_array_ptr)[Py::make_tuple(dim_i, sample_i)] ) ;
      loss_tmp = calc_loss(th_tmp, dim_i);
      
      if (loss_tmp < loss)// record
      {
        loss = loss_tmp;
        threshold = th_tmp;
        feature_idx = dim_i;
        // record the count...
        std::copy(label_R_cnt_ptr.get(), label_R_cnt_ptr.get()+class_num, loss_R_cnt_ptr.get());
        std::copy(label_L_cnt_ptr.get(), label_L_cnt_ptr.get()+class_num, loss_L_cnt_ptr.get());
      }

    }
  }
  // call tags
  set_tags();

}

float Node::calc_loss(float th, int dim_i)
{

  for(int class_i=0; class_i < class_num;++class_i)
  {
    label_L_cnt_ptr.get()[class_i] = 0.;
    label_R_cnt_ptr.get()[class_i] = 0.;
  }
  /* calculate the purity */
  int list_len = Py::len(*sample_list_ptr);
  for(int list_i = 0; list_i < list_len; ++list_i)
  {
    int sample_i = int( Py::extract<float>((*sample_list_ptr)[list_i]) );
    if ( (*sample_array_ptr)[Py::make_tuple(dim_i, sample_i)] < th ) // count into left
      ++(label_L_cnt_ptr.get()[ int( Py::extract<float>( (*sample_array_ptr)[Py::make_tuple(feature_dims, sample_i)]) ) ]) ;
    else
      ++(label_R_cnt_ptr.get()[int(Py::extract<float>( (*sample_array_ptr)[Py::make_tuple(feature_dims, sample_i)]) ) ]) ;
  }
  // the loss...
  float label_L_cnt, label_R_cnt;
  label_L_cnt=label_R_cnt=0.;

  for(int class_i=0; class_i < class_num;++class_i)
  {
     label_L_cnt +=  label_L_cnt_ptr.get()[class_i] ;
     label_R_cnt +=  label_R_cnt_ptr.get()[class_i] ;
  }

  for(int class_i=0; class_i < class_num;++class_i)
  {
     label_L_cnt_ptr.get()[class_i] /=label_L_cnt ;
     label_R_cnt_ptr.get()[class_i] /=label_R_cnt ;
  }
//  float loss_l, loss_r;
//  loss_l = loss_r=0.;
 float loss_loc=0.;
  /*
        USE entropy
  */
  for(int class_i=0; class_i < class_num;++class_i)
  {
    if (label_L_cnt_ptr.get()[class_i] > postive_zero)
      loss_loc += -label_L_cnt_ptr.get()[class_i] *log(label_L_cnt_ptr.get()[class_i]) ;
    if (label_R_cnt_ptr.get()[class_i] > postive_zero)
      loss_loc += -label_R_cnt_ptr.get()[class_i] *log(label_R_cnt_ptr.get()[class_i]) ;
  } 
  return loss_loc;



}

Py::list Node::predict(Py::numeric::array &feature)
{
  if(Py::len(left_labels)<1 && Py::len(right_labels)<1 )
  {
    LOG(FATAL)<<"This node has not been trained!";
    throw -1;
  }
  Py::list ret;
  Py::tuple shape =Py::extract<Py::tuple>(feature.attr("shape"));
  int num = Py::extract<int>( shape[1]   );
  for(int sample_i=0; sample_i<num; ++sample_i)
  {
    if( Py::extract<float>(feature[Py::make_tuple(feature_idx, sample_i)] ) < threshold  )// left
      ret.append(left_labels);
    else
      ret.append(right_labels);
  }
  return ret;
}


void Node::set_tags()
{
/*
  for(int list_i=0; sample_i<list_len; ++sample)
  {
    int sample_i = int(Py::extract<float>( (*sample_list_ptr)[list_i]  )  );
    if ( Py::extract<float>( sample_array[Py::make_tuple(feature_idx,sample_i)]) < threshold   ) //
  }
*/

  
  Py::list left_labels_tmp, right_labels_tmp;
  for(int i=0; i<class_num;++i)
  {
    if (loss_L_cnt_ptr.get()[i] > postive_zero)// push into left
      left_labels_tmp.append(i);
    else
      right_labels_tmp.append(i);

  }

  left_labels  = left_labels_tmp;
  right_labels = right_labels_tmp;

}
std::string myname()
{
  return "my name is Chen Yliang";
}
using namespace boost::python;
BOOST_PYTHON_MODULE(node_test)
{
  numeric::array::set_module_and_type("numpy", "ndarray");
//  def("myname", &myname);

  class_<Node>("Node",init<Node &>())
  .def(init<Py::list &> ())
  .def(init<Py::numeric::array &,Py::list & , int>())

  .def("train",&Node::train)
  .def("predict", &Node::predict)

   .def_readwrite("feature_dims",&Node::feature_dims)
  .def_readwrite("sample_array",&Node::sample_array_ptr)
  .def_readwrite("sample_list",&Node::sample_list_ptr)
  .def_readwrite("class_num",&Node::class_num)
 .def_readonly("threshold",&Node::threshold)

  .def_readonly("loss",&Node::loss)
  .def_readonly("feature_idx",&Node::feature_idx)
  .def_readonly("labels",&Node::labels)
  .def_readonly("right_labels", &Node::right_labels)
  .def_readonly("left_labels", &Node::left_labels);
}

