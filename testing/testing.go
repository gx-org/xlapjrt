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

// Package testing provide a cgx runtime for GX testing.
// The runtime includes GX test files statically linked in the binary.
package testing

import (
	"github.com/gx-org/xlapjrt/plugin"
	"github.com/gx-org/gx/api"
	"github.com/gx-org/gx/cgx/handle"
	gxtesting "github.com/gx-org/gx/tests/testing"
	pjrtstdlib "github.com/gx-org/xlapjrt/stdlib"
)

// #include <gxdeps/github.com/gx-org/gx/golang/binder/cgx/cgx.h>
import "C"

//export cgx_testing_runtime
func cgx_testing_runtime() C.struct_cgx_runtime_new_result {
	bld := gxtesting.NewBuilderStaticSource(pjrtstdlib.Stdlib)
	rtm, err := plugin.NewWithBuilder("cpu", bld)
	if err != nil {
		return C.struct_cgx_runtime_new_result{
			error: (C.cgx_error)(handle.Wrap[error](err)),
		}
	}
	return C.struct_cgx_runtime_new_result{
		runtime: C.cgx_runtime(handle.Wrap[*api.Runtime](rtm)),
	}
}
