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
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend/dtype"
	"github.com/gx-org/backend/shape"
)

// Supported are the types supported by this backend.
type Supported interface {
	dtype.GoDataType
	dtypes.Supported
}

// ToGXDType converts a gopjrt DType to a GX datatype.
func ToGXDType(k dtypes.DType) dtype.DataType {
	switch k {
	case dtypes.Bool:
		return dtype.Bool
	case dtypes.Float32:
		return dtype.Float32
	case dtypes.Float64:
		return dtype.Float64
	case dtypes.Int32:
		return dtype.Int32
	case dtypes.Int64:
		return dtype.Int64
	case dtypes.Uint32:
		return dtype.Uint32
	case dtypes.Uint64:
		return dtype.Uint64
	}
	return dtype.Invalid
}

// ToDType converts a GX kind into a gopjrt DType.
func ToDType(k dtype.DataType) dtypes.DType {
	switch k {
	case dtype.Bool:
		return dtypes.Bool
	case dtype.Float32:
		return dtypes.Float32
	case dtype.Float64:
		return dtypes.Float64
	case dtype.Int32:
		return dtypes.Int32
	case dtype.Int64:
		return dtypes.Int64
	case dtype.Uint32:
		return dtypes.Uint32
	case dtype.Uint64:
		return dtypes.Uint64
	}
	return dtypes.InvalidDType
}

// ToShape converts a GX shape into a gopjrt/xla shape.
func ToShape(shape *shape.Shape) xlabuilder.Shape {
	return xlabuilder.Shape{
		DType:      ToDType(shape.DType),
		Dimensions: shape.AxisLengths,
	}
}

// ToGXShape converts a gopjrt shape into a GX shape.
func ToGXShape(sh xlabuilder.Shape) *shape.Shape {
	return &shape.Shape{
		DType:       ToGXDType(sh.DType),
		AxisLengths: sh.Dimensions,
	}
}
