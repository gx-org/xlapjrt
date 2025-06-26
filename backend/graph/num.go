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
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend/ops"
	"github.com/gx-org/backend/shape"
	pjrtgx "github.com/gx-org/xlapjrt"
)

// Num returns the builder to build operations from the num package.
func (g *Graph) Num() ops.NumBuilder {
	return g
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0) on the given axis.
func (g *Graph) Iota(shape *shape.Shape, iotaAxis int) (ops.Node, error) {
	xlaOp, err := xlabuilder.Iota(g.builder, pjrtgx.ToShape(shape), iotaAxis)
	if err != nil {
		return nil, err
	}
	return g.newNode(xlaOp), nil
}
