#ifndef FSM_STATE_RLJOINTPD_H
#define FSM_STATE_RLJOINTPD_H

#include "FSM_State.h"
#include <cstring>
#include <experimental/filesystem>
#include "cpuMLP.hpp"
#include "../common/include/Controllers/StateEstimatorContainer.h"
#include "../common/include/Controllers/LegController.h"
#define OBSDIM 141  // TODO
#define UNOBSDIM 11  // TODO

/**
 *
 */
template <typename T>
class FSM_State_RLJointPD : public FSM_State<T> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FSM_State_RLJointPD(ControlFSMData<T>* _controlFSMData);

  // Behavior to be carried out when entering a state
  void onEnter();

  // Run the normal behavior for the state
  void run();

  // Checks for any transition triggers
  FSM_StateName checkTransition();

  // Manages state specific transitions
  TransitionData<T> transition();

  // Behavior to be carried out when exiting a state
  void onExit();

  virtual void updateHistory();
  virtual void updateObservation();
  virtual void updatePreviousActions();
  virtual const Eigen::Matrix<float, OBSDIM, 1>& getObservation();

 private:
  // Keep track of the control iterations
  int iter = 0;
  DVec<T> _ini_jpos;

 protected:

  float _bodyHeight;
  Eigen::Matrix<float, 3, 1> _bodyOri;
  Eigen::Matrix<float, 12, 1> _jointQ;
  Eigen::Matrix<float, 3, 1> _bodyVel;
  Eigen::Matrix<float, 3, 1> _bodyAngularVel;
  Eigen::Matrix<float, 12, 1> _jointQd;
  Eigen::Matrix<float, OBSDIM, 1> _obs;
  Eigen::Matrix<float, OBSDIM + UNOBSDIM, 1> actorObs_;
  Eigen::Matrix<float, UNOBSDIM, 1> estOut_;
  int _obsDim, historyLength_, nJoints_, actionDim_;
  Eigen::VectorXf _obsMean, _obsVar;
  Eigen::VectorXf jointPosErrorHist_, jointVelHist_, historyTempMem_;
  Eigen::VectorXf q_init;
  Eigen::VectorXf pTarget12_, pTarget12_prev_;
  Eigen::VectorXf previousAction_, prepreviousAction_;
  Eigen::VectorXf footPos_;
  Eigen::Vector3f command_;

  std::string _loadPath;

  std::chrono::steady_clock::time_point begin_;
  std::chrono::steady_clock::time_point end_;

  rai::FuncApprox::MLP_fullyconnected<float, OBSDIM + UNOBSDIM, 12, rai::FuncApprox::ActivationType::leakyrelu> policy;
  rai::FuncApprox::MLP_fullyconnected<float, OBSDIM, UNOBSDIM, rai::FuncApprox::ActivationType::leakyrelu> estimator;
  double control_dt_ = 0.01;

  bool emergency_stop = false;

  LegControllerCommand<float> preCommands[4];
};

#endif  // FSM_STATE_RLJOINTPD_H
