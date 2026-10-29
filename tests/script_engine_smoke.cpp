#include "Utils/MatlabInterface.h"
#include "MatFile.h"
#include <algorithm>
#include <vector>
#include <complex>
#include <filesystem>
#include <iostream>
#ifdef BCP_OCTAVE_ENGINE
#include <chrono>
#include <thread>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

static void check(bool result, const char* message) {
    if (!result) throw std::runtime_error(message);
}
int main(int argc, char** argv) try {
#ifdef BCP_OCTAVE_ENGINE
    // popen2 passes the worker script as argv[1]; simulate a worker that hangs.
    if (argc == 2) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        return 1;
    }
#endif
#ifdef BCP_SKIP_SCRIPT_TEST
    std::cout << "Numerical script runtime unavailable; test not run\\n";
    return 77;
#endif
    auto& engine=MatlabInterface::GetEngine();
    engine.AddScriptPath(BCP_MATLAB_SCRIPTS_DIR);
    engine.Eval("gpu_policy_test = double(bcp_gpu_enabled('cpu'));");
    double enabled = 1;
    check(!engine.GetEngineRealMatrix("gpu_policy_test",1,1,&enabled) && enabled==0,
          "Explicit GIF CPU policy was not honored");
    double input[]={1,2,3,4,5,6}, output[6]={};
    check(!engine.SetEngineRealMatrix("GIF.test",2,3,input),"Dense set failed");
    engine.Eval("GIF.test = GIF.test * 2;");
    unsigned int rows=0,cols=0;
    check(engine.GetMatrixDimensions("GIF.test",rows,cols) && rows==2 && cols==3,"Shape mismatch");
    check(!engine.GetEngineRealMatrix("GIF.test",2,3,output),"Dense get failed");
    for(int i=0;i<6;++i) check(output[i]==2*input[i],"Dense layout mismatch");
    unsigned int ri[]={0,1}, ci[]={1,0};
    std::complex<double> values[]={{2,3},{4,-1}}, dense[4];
    check(!engine.SetEngineSparseComplexMatrix("s",2,ri,ci,values,2,2),"Sparse set failed");
    engine.Eval("s=full(s);");
    check(!engine.GetEngineComplexMatrix("s",2,2,dense),"Complex get failed");
    check(dense[1]==values[0] && dense[2]==values[1],"Sparse complex mismatch");
    std::vector<std::complex<double>> large(8193*2), roundtrip(large.size());
    for (size_t i=0;i<large.size();++i) large[i]={double(i)/7.0,-double(i)/11.0};
    check(!engine.SetEngineComplexMatrix("large",8193,2,large.data()),"Large complex set failed");
    check(!engine.GetEngineComplexMatrix("large",8193,2,roundtrip.data()) && large==roundtrip,
          "Large MAT transfer changed values");
    engine.Eval("clear;");
    check(!engine.SetEngineRealMatrix("a",2,3,input),"Transfer after clear failed");
    engine.Eval("clearvars -except a; a = a + 1;");
    check(!engine.GetEngineRealMatrix("a",2,3,output),"Transfer after clearvars failed");
    check(output[0]==2 && output[5]==7,"Workspace persistence failed");
    bool caught=false;
    try { engine.Eval("fprintf(1, '%s', repmat('x',1,4097)); error('intentional smoke error');"); }
    catch(const std::exception&) { caught=true; }
    check(caught,"Script errors were silently ignored");
#ifdef BCP_OCTAVE_ENGINE
    for (int count : {4064,4080,4095,4096,4097,MAX_OUTPUT_BUFFER_SIZE+4096}) {
        const auto code="fprintf(1, '%s', repmat('x',1,"+std::to_string(count)+"));";
        const auto text=engine.EvalToString(code.c_str(),false);
        check(text==std::string(std::min(count,MAX_OUTPUT_BUFFER_SIZE-1),'x'),
              "Chunked output or sentinel framing changed");
    }
    const auto executable=std::filesystem::canonical(argv[0]).string();
    const char* worker=executable.c_str();
    check(!engine.SetEngineStringArray("worker_stub",1,&worker),"Cannot set worker stub");
    engine.Eval(
        "saved_python = getenv('BCP_PYTHON_EXECUTABLE'); saved_timeout = getenv('BCP_TORCH_TIMEOUT_SECONDS');"
        "setenv('BCP_PYTHON_EXECUTABLE',worker_stub); setenv('BCP_TORCH_TIMEOUT_SECONDS','1');"
        "timeout_seen = false; try; bcp_torch_energy('start','cpu'); catch failure;"
        "assert(strcmp(failure.identifier,'BCP:TorchTimeout'),failure.message); timeout_seen = true; end;"
        "assert(timeout_seen); setenv('BCP_PYTHON_EXECUTABLE',saved_python);"
        "setenv('BCP_TORCH_TIMEOUT_SECONDS',saved_timeout);");
    caught=false;
    try { engine.Eval("bcp_gpu_enabled('gpu');"); }
    catch(const std::exception&) { caught=true; }
    check(caught,"Octave silently accepted a MATLAB GPU request");
    engine.Eval("nd_test = zeros(2,2,2);");
    caught=false;
    try { engine.GetEngineRealMatrix("nd_test",2,2,output); }
    catch(const std::exception&) { caught=true; }
    check(caught,"Octave silently truncated a three-dimensional matrix");

    engine.Eval("engine_test_pid = double(getpid());");
    double octave_pid=0;
    check(!engine.GetEngineRealMatrix("engine_test_pid",1,1,&octave_pid),"Cannot read engine PID");
    const auto pid=static_cast<pid_t>(octave_pid);
    check(pid>1 && pid!=getpid(),"Invalid engine PID");
    check(kill(pid,SIGKILL)==0,"Cannot stop test engine");
    int status=0;
    check(waitpid(pid,&status,0)==pid,"Cannot reap test engine");
    caught=false;
    try { MatlabInterface::GetEngine(); } catch(const std::exception&) { caught=true; }
    check(caught,"Dead engine was silently restarted by GetEngine");
    caught=false;
    try { engine.Eval("unexpected_restart = 1;"); } catch(const std::exception&) { caught=true; }
    check(caught,"Dead engine was silently restarted by Eval");
    caught=false;
    try { engine.GetMatrixDimensions("a",rows,cols); } catch(const std::exception&) { caught=true; }
    check(caught,"Dead engine was silently restarted by dimension query");
    auto& restarted=MatlabInterface::GetEngine(true);
    restarted.Eval("explicit_restart_ok = double(isempty(who('a')));");
    double fresh=0;
    check(!restarted.GetEngineRealMatrix("explicit_restart_ok",1,1,&fresh) && fresh==1,
          "Explicit engine restart did not create a fresh workspace");
#endif
    std::cout << "Script engine numerical tests passed\n";
    return 0;
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
