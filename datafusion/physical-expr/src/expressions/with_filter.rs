// Copyright 2022 poonai
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

use crate::PhysicalExpr;
use arrow::compute::filter_record_batch;
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use std::sync::Arc;
use arrow::error::Result as ArrowResult;
use std::fmt;
use arrow::array::BooleanArray;

#[derive(Debug)]
pub struct ExprWithFilter {
    expr: Arc<dyn PhysicalExpr>,
    filter: Arc<dyn PhysicalExpr>,
}

impl ExprWithFilter {
    pub fn new(expr: Arc<dyn PhysicalExpr>, filter: Arc<dyn PhysicalExpr>) -> Self {
        Self{
            expr: expr,
            filter: filter
        }
    }
}

impl fmt::Display for ExprWithFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} FILTER (WHERE {})", self.expr, self.filter)
    }
}

impl PhysicalExpr for ExprWithFilter {
    fn as_any(&self) -> &dyn std::any::Any {
        self.expr.as_any()
    }

    fn data_type(
        &self,
        input_schema: &arrow::datatypes::Schema,
    ) -> datafusion_common::Result<arrow::datatypes::DataType> {
        self.expr.data_type(input_schema)
    }

    fn nullable(
        &self,
        input_schema: &arrow::datatypes::Schema,
    ) -> datafusion_common::Result<bool> {
        self.expr.nullable(input_schema)
    }

    fn evaluate(
        &self,
        batch: &RecordBatch,
    ) -> datafusion_common::Result<datafusion_expr::ColumnarValue> {
        let tmp_batch = batch_filter(batch, &self.filter)?;
        self.expr.evaluate(&tmp_batch)
    }

}

pub fn batch_filter(
    batch: &RecordBatch,
    predicate: &Arc<dyn PhysicalExpr>,
) -> ArrowResult<RecordBatch> {
    predicate
        .evaluate(batch)
        .map(|v| v.into_array(batch.num_rows()))
        .map_err(DataFusionError::into)
        .and_then(|array| {
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "Filter predicate evaluated to non-boolean value".to_string(),
                    )
                    .into()
                })
                // apply filter array to record batch
                .and_then(|filter_array| filter_record_batch(batch, filter_array))
        })
}
