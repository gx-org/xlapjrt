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

// Package xlapjrt provides a gopjrt backend to GX.
package xlapjrt

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/shapes"
	pjtypes "github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/xlabuilder"
)

// Supported are the types supported by this backend.
type Supported interface {
	gotype.Supported
	pjtypes.Supported
}

// ToGXDType converts a gopjrt DType to a GX datatype.
func ToGXDType(k pjtypes.DType) dtypes.DType {
	switch k {
	case pjtypes.Bool:
		return dtypes.Bool
	case pjtypes.BFloat16:
		return dtypes.BFloat16
	case pjtypes.Float32:
		return dtypes.Float32
	case pjtypes.Float64:
		return dtypes.Float64
	case pjtypes.Int32:
		return dtypes.Int32
	case pjtypes.Int64:
		return dtypes.Int64
	case pjtypes.Uint32:
		return dtypes.Uint32
	case pjtypes.Uint64:
		return dtypes.Uint64
	}
	return dtypes.InvalidDType
}

// ToPJDType converts a GX kind into a gopjrt DType.
func ToPJDType(k dtypes.DType) pjtypes.DType {
	switch k {
	case dtypes.Bool:
		return pjtypes.Bool
	case dtypes.BFloat16:
		return pjtypes.BFloat16
	case dtypes.Float32:
		return pjtypes.Float32
	case dtypes.Float64:
		return pjtypes.Float64
	case dtypes.Int32:
		return pjtypes.Int32
	case dtypes.Int64:
		return pjtypes.Int64
	case dtypes.Uint32:
		return pjtypes.Uint32
	case dtypes.Uint64:
		return pjtypes.Uint64
	}
	return pjtypes.InvalidDType
}

// ToShape converts a GX shape into a gopjrt/xla shape.
func ToShape(shape shapes.Shape) xlabuilder.Shape {
	return xlabuilder.Shape{
		DType:      ToPJDType(shape.DType),
		Dimensions: shape.Dimensions,
	}
}

// ToGXShape converts a gopjrt shape into a GX shape.
func ToGXShape(sh xlabuilder.Shape) shapes.Shape {
	return shapes.Make(ToGXDType(sh.DType), sh.Dimensions...)
}
