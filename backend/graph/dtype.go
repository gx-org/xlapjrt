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

package graph

import (
	"github.com/pkg/errors"
	"github.com/gomlx/compute/dtypes"
	pjtypes "github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend"
	pjrtgx "github.com/gx-org/xlapjrt"
)

// Bitcast returns a bitcast/reinterpret operator node.
func (g *Graph) Bitcast(x backend.Value, target dtypes.DType) (backend.Value, error) {
	xlaDType := pjrtgx.ToPJDType(target)
	if xlaDType == pjtypes.InvalidDType {
		return nil, errors.Errorf("cannot convert %s to a XLA data type", target.String())
	}
	xlaOp, err := xlabuilder.Bitcast(g.xlaHandle(x), xlaDType)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}
