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
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/xlabuilder"
	dtype "github.com/gx-org/backend/dtypes"
	"github.com/gx-org/backend/ops"
	pjrtgx "github.com/gx-org/xlapjrt"
)

// DType returns the builder to build operations from the dtype package.
func (g *Graph) DType() ops.DTypeBuilder {
	return g
}

// Bitcast returns a bitcast/reinterpret operator node.
func (g *Graph) Bitcast(x ops.Node, target dtype.DType) (ops.Node, error) {
	xlaDType := pjrtgx.ToDType(target)
	if xlaDType == dtypes.InvalidDType {
		return nil, errors.Errorf("cannot convert %s to a XLA data type", target.String())
	}
	xlaOp, err := xlabuilder.Bitcast(g.xlaHandle(x), xlaDType)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}
