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

package bindings_test

import (
	"testing"

	"github.com/gx-org/xlapjrt/plugin"
	bindingstests "github.com/gx-org/gx/golang/tests"
	gxtesting "github.com/gx-org/gx/tests/testing"
	pjrtstdlib "github.com/gx-org/xlapjrt/stdlib"
)

func TestGoBindings(t *testing.T) {
	bld := gxtesting.NewBuilderStaticSource(pjrtstdlib.Stdlib)
	rtm, err := plugin.NewWithBuilder("cpu", bld)
	if err != nil {
		t.Fatalf("\n%+v", err)
	}
	bindingstests.RunAll(t, rtm)
}
