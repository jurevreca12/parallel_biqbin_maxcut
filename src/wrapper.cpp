#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "biqbin_cpp_api.h"
#include "blas_laplack.h"

#include "wrapper.h"

namespace py = pybind11;

/* biqbin's global variables from global_var.h */
extern Problem *SP;
extern Problem *PP;
extern BabSolution *BabSol;
extern int BabPbSize;

/* final solution */
std::vector<int> selected_nodes;
double running_time;
int rank;

// Global Python override function (if any)
py::object python_heuristic_override;
py::object py_read_data_override;

/// @brief set heuristic function from Python
/// @param func
void set_heuristic_override(py::object func)
{
    python_heuristic_override = func;
}

/// @brief set instance reading function from Python
/// @param func
void set_read_data_override(py::object func)
{
    py_read_data_override = func;
}

void set_rank(int r)
{
    rank = r;
}

int get_rank()
{
    return rank;
}

/// @brief TODO: find a better fix for conflicts with MPI
void clean_python_references(void)
{
    py_read_data_override = py::object(); // Clear the callback
    python_heuristic_override = py::object();
}

/// @brief Helper functions for better error messages
/// @tparam T int or double
/// @return string of type T
template <typename T>
const char *type_name();
template <>
const char *type_name<double>() { return "float64"; }
template <>
const char *type_name<int>() { return "int32"; }

/// @brief Checks whether c++ is getting the correct format numpy array from Python, throws error
/// @tparam T either a double or int
/// @param np_in numpy array passed in
/// @param dimensions checks the shape of the np array
//template <typename T>
//void check_np_array_validity(const py::array_t &np_in, int dimensions, const std::string &np_array_name = "")
//{
//    // Check dtype
//    if (np_in.get_dtype() != np::dtype::get_builtin<T>())
//    {
//        std::string msg = np_array_name + " - Incorrect array data type: expected " + std::string(type_name<T>());
//        PyErr_SetString(PyExc_TypeError, msg.c_str());
//        p::throw_error_already_set();
//    }
//    // Check number of dimensions
//    if (np_in.get_nd() != dimensions)
//    {
//        std::string msg = np_array_name + " - Incorrect number of dimensions: expected " +
//                          std::to_string(dimensions) + ", got " + std::to_string(np_in.get_nd());
//        PyErr_SetString(PyExc_TypeError, msg.c_str());
//        p::throw_error_already_set();
//    }
//
//    // If 2D, check for square shape
//    if (dimensions == 2 && np_in.shape(0) != np_in.shape(1))
//    {
//        std::string msg = np_array_name + " - Incorrect shape: expected a square (n x n) array, got a (" +
//                          std::to_string(np_in.shape(0)) + " x " + std::to_string(np_in.shape(1)) + " )";
//        PyErr_SetString(PyExc_ValueError, msg.c_str());
//        p::throw_error_already_set();
//    }
//
//    // Check row-major contiguous
//    if (!(np_in.get_flags() & np::ndarray::C_CONTIGUOUS))
//    {
//        std::string msg = np_array_name + " - Array must be row-major contiguous";
//        PyErr_SetString(PyExc_TypeError, msg.c_str());
//        p::throw_error_already_set();
//    }
//}

/// @brief Creates a numpy array of the solution, returned after biqbin is done solving
/// @return np.ndarray(dtype = np.int32) of the final solution (node names in a np list)
py::array_t<int> get_selected_nodes_np_array()
{
    ssize_t              ndim    = 1;
    std::vector<long unsigned int> shape   = {selected_nodes.size()};
    std::vector<ssize_t> strides = {sizeof(int)};

    py::array_t<int> result(py::buffer_info(
        selected_nodes.data(),
        sizeof(int),
        py::format_descriptor<int>::format(),
        ndim,
        shape,
        strides
    ));
    return result;
}

/// @brief
/// @param py_args argv in a Python string list ["biqbin", "graph_instance_path", "parameters_path"]
/// @return dictionary of "max_val" value of maximum cut and "solution" vertices

/// @brief Run the solver, retrieve the solution
/// @param prog_name argv[0] "biqbin_*.py"
/// @param problem_instance_name argv[1] "problem_path_to_file"
/// @param params_file_name argv[2] "path_to_params_file"
/// @return biqbin maxcut result
py::dict run_py(char *prog_name, char *problem_instance_name, char *params_file_name)
{
    char *argv[3] = {prog_name, problem_instance_name, params_file_name};

    wrapped_main(3, argv);
    clean_python_references(); // TODO: handle python references better

    // Build result dictionary
    py::dict result_dict, nested;
    result_dict["time"] = running_time;
    result_dict["maxcut"] = nested;
    result_dict["maxcut"]["computed_val"] = Bab_LBGet();
    result_dict["maxcut"]["solution"] = get_selected_nodes_np_array();
    return result_dict;
}

double run_heuristic_python(
    const py::array &P0_L_array,
    const py::array &P_L_array,
    const py::array &xfixed_array,
    const py::array &node_sol_X_array,
    const py::array &x_array)
{
    // Check if input is valid
    //check_np_array_validity<double>(P0_L_array, 2, "P0_L");
    //check_np_array_validity<double>(P_L_array, 2, "P_L");
    //check_np_array_validity<int>(xfixed_array, 1, "xfixed");
    //check_np_array_validity<int>(node_sol_X_array, 1, "node_sol_x");
    //check_np_array_validity<int>(x_array, 1, "x");

    double *P0_L = reinterpret_cast<double *>(P0_L_array.ptr());
    double *P_L = reinterpret_cast<double *>(P_L_array.ptr());
    int *xfixed = reinterpret_cast<int *>(xfixed_array.ptr());
    int *node_sol_X = reinterpret_cast<int *>(node_sol_X_array.ptr());
    int *x = reinterpret_cast<int *>(x_array.ptr());

    return runHeuristic_unpacked(P0_L, P0_L_array.shape(0), P_L, P_L_array.shape(0), xfixed, node_sol_X, x);
}

/// @brief Called in runHeuristic in heuristic.c
/// @param P0 is the original Problem *SP in global_var.h
/// @param P  current subproblem Problem *PP in global_var.h
/// @param node current branch and bound node
/// @param x stores the best solution nodes found the by the heuristic function
/// @return best lower bound of the current subproblem found by the heuristic used
double wrapped_heuristic(Problem *P0, Problem *P, BabNode *node, int *x)
{ 
    py::array_t<double> P0_L_array(py::buffer_info(
        P0->L,
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        {P0->n, P0->n},
        {sizeof(double) * P0->n, sizeof(double)}
    ));

    py::array_t<double> P_L_array = py::array(py::buffer_info(
        P->L,
        sizeof(double),
        py::format_descriptor<double>::format(),
        2, // ndim
        {P->n, P->n},
        {sizeof(double) * P->n, sizeof(double)}
    ));

    py::array_t<int> xfixed_array = py::array(py::buffer_info(
        node->xfixed,
        sizeof(int),
        py::format_descriptor<int>::format(),
        1, // ndim
        {P0->n},
        {sizeof(int)}
    ));

    py::array_t<int> sol_X_array = py::array(py::buffer_info(
        node->sol.X,
        sizeof(int),
        py::format_descriptor<int>::format(),
        1, // ndim
        {P0->n - 1},
        {sizeof(int)}
    ));

    py::array_t<int> x_array = py::array(py::buffer_info(
        x,
        sizeof(int),
        py::format_descriptor<int>::format(),
        1, // ndim
        {BabPbSize},
        {sizeof(int)}
    ));

    // RK https://wiki.python.org/moin/boost.python/extract
    py::object heuristic_val = python_heuristic_override(
        P0_L_array, 
        P_L_array, 
        xfixed_array, 
        sol_X_array,
        x_array
    );
    return heuristic_val.cast<double>(); 
}

/// @brief Read the instance problem file return the adjacency matrix
/// @param instance path to instance file
/// @return adjacency matrix
py::array_t<double> read_data_python(const char *instance)
{
    double *adj;
    int adj_N;
    adj = readData(instance, &adj_N); // readData exits the program if parsing fails.
    
    ssize_t              ndim    = 2;
    std::vector<ssize_t> shape   = {adj_N, adj_N};
    std::vector<long unsigned int> strides = {sizeof(double) * adj_N, sizeof(double)};

    py::array_t<double> array(py::buffer_info(
        adj,
        sizeof(double),
        py::format_descriptor<double>::format(),
        ndim,
        shape,
        strides
    ));
    return array;
}

/// @brief Get an adjacency matrix from Python and set Problem *SP->L and *PP global variables
int wrapped_read_data()
{
    // Declaration of np_adj before the try-catch, in case Python threw an error
    // https://live.boost.org/doc/libs/1_80_0/libs/python/doc/html/numpy/tutorial/ndarray.html
    py::array np_adj = py_read_data_override();
    //try
    //{
    //    np_adj = py::extract<py::array_t>(py_read_data_override());
    //}
    //catch (const boost::python::error_already_set&)
    //{
    //    PyErr_Print();
    //    std::exit(1);
    //}
    //check_np_array_validity<double>(np_adj, 2, "adj");
    int adj = process_adj_matrix(
        reinterpret_cast<double *>(np_adj.ptr()),
        np_adj.shape(0)
    );
    return adj;
}

// Python module exposure
PYBIND11_MODULE(biqbin, m)
{
    m.doc() = "The biqbin binary optimization package.";

    m.def("set_heuristic", &set_heuristic_override, "Override the heuristics used.");
    m.def("set_read_data", &set_read_data_override, "Set read data.");
    m.def("read_bqp_data", &read_data_BQP, "Read BQP.");
    m.def("run", &run_py, "Run blabla.");
    m.def("default_heuristic", &run_heuristic_python, "Get default heuristic.");
    m.def("default_read_data", &read_data_python, "Read default data.");
    m.def("get_rank", &get_rank, "Get the rank.");
}

/// @brief Copy the solution before memory is freed, so it can be retrieved in Python // RK ???
void copy_solution()
{
    for (int i = 0; i < BabPbSize; ++i) // RK I need to you Beno to explain me this !!!
    {
        if (BabSol->X[i] == 1)
        {
            selected_nodes.push_back(i + 1); // 1-based indexing
        }
    }
}

/// @brief record time at the end
/// @param time
void record_time(double time)
{
    running_time = time;
}
