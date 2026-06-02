// Copyright 2021 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "envpool/core/py_envpool.h"
#include "envpool/vizdoom/vizdoom_env.h"

using VizdoomEnvSpec = PyEnvSpec<vizdoom::VizdoomEnvSpec>;

class PyVizdoomEnvPool : public PyEnvPool<vizdoom::VizdoomEnvPool> {
 public:
  using PyEnvPool<vizdoom::VizdoomEnvPool>::PyEnvPool;

  py::array_t<uint8_t> PyTerminalResetObs(const py::array& env_ids) {
    py::array_t<int, py::array::c_style | py::array::forcecast> env_ids_array(
        env_ids);
    auto env_ids_view = env_ids_array.unchecked<1>();
    std::vector<py::ssize_t> shape;
    shape.emplace_back(env_ids_view.shape(0));
    const auto& obs_shape = this->envs_[0]->TerminalResetObsShape();
    for (auto dim : obs_shape) {
      shape.emplace_back(static_cast<py::ssize_t>(dim));
    }
    py::array_t<uint8_t> out(shape);
    auto* dst = static_cast<uint8_t*>(out.mutable_data());
    std::size_t obs_size = this->envs_[0]->TerminalResetObsSize();
    {
      py::gil_scoped_release release;
      for (py::ssize_t i = 0; i < env_ids_view.shape(0); ++i) {
        this->envs_[env_ids_view(i)]->CopyTerminalResetObsTo(
            dst + i * obs_size);
      }
    }
    return out;
  }
};

PYBIND11_MODULE(vizdoom_envpool, m) {
  py::class_<VizdoomEnvSpec>(m, "_VizdoomEnvSpec", py::metaclass(abc_meta))
      .def(py::init<const VizdoomEnvSpec::ConfigValues&>())
      .def_readonly("_config_values", &VizdoomEnvSpec::py_config_values)
      .def_readonly("_state_spec", &VizdoomEnvSpec::py_state_spec)
      .def_readonly("_action_spec", &VizdoomEnvSpec::py_action_spec)
      .def_readonly_static("_state_keys", &VizdoomEnvSpec::py_state_keys)
      .def_readonly_static("_action_keys", &VizdoomEnvSpec::py_action_keys)
      .def_readonly_static("_config_keys", &VizdoomEnvSpec::py_config_keys)
      .def_readonly_static("_default_config_values",
                           &VizdoomEnvSpec::py_default_config_values);
  py::class_<PyVizdoomEnvPool>(m, "_VizdoomEnvPool", py::metaclass(abc_meta))
      .def(py::init<const VizdoomEnvSpec&>())
      .def_readonly("_spec", &PyVizdoomEnvPool::py_spec)
      .def("_recv", &PyVizdoomEnvPool::PyRecv)
      .def("_send", &PyVizdoomEnvPool::PySend)
      .def("_reset", &PyVizdoomEnvPool::PyReset)
      .def("_terminal_reset_obs", &PyVizdoomEnvPool::PyTerminalResetObs)
      .def_readonly_static("_state_keys", &PyVizdoomEnvPool::py_state_keys)
      .def_readonly_static("_action_keys", &PyVizdoomEnvPool::py_action_keys)
      .def("_xla", &PyVizdoomEnvPool::Xla);
}
