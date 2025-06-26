// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package cgx provides CGX access to the XLA-PJRT backend.
package cgx

import (
	"github.com/gx-org/xlapjrt/plugin"
	"github.com/gx-org/gx/api"
	"github.com/gx-org/gx/build/builder"
	"github.com/gx-org/gx/build/importers/embedpkg"
	"github.com/gx-org/gx/cgx/handle"
	pjrtstdlib "github.com/gx-org/xlapjrt/stdlib"
)

// #include <gx/golang/binder/cgx/cgx.h>
import "C"

//export cgx_builder_new_static_xlapjrt
func cgx_builder_new_static_xlapjrt() C.cgx_builder {
	return C.cgx_builder(handle.Wrap[*builder.Builder](embedpkg.NewBuilder(pjrtstdlib.Stdlib)))
}

//export cgx_runtime_new_xlapjrt
func cgx_runtime_new_xlapjrt(cbld C.cgx_builder, cPluginName *C.cchar_t) C.struct_cgx_runtime_new_result {
	bld := handle.Unwrap[*builder.Builder](handle.Handle(uintptr(cbld)))
	rtm, err := plugin.NewWithBuilder(C.GoString(cPluginName), bld)
	return C.struct_cgx_runtime_new_result{
		runtime: (C.cgx_runtime)(handle.Wrap[*api.Runtime](rtm)),
		error:   (C.cgx_error)(handle.Wrap[error](err)),
	}
}
