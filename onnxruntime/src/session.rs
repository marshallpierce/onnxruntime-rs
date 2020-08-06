use std::{
    ffi::CString,
    path::PathBuf,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc, Mutex,
    },
};

use onnxruntime_sys as sys;

use crate::{
    char_p_to_string,
    env::{Env, NamedEnv},
    error::{status_to_result, OrtError, Result},
    g_ort, GraphOptimizationLevel, TensorElementDataType,
};

// FIXME: Create a high-level wrapper
pub struct SessionOptions {
    ptr: *mut sys::OrtSessionOptions,
}

pub struct Session {
    session_ptr: *mut sys::OrtSession,
    allocator_ptr: *mut sys::OrtAllocator,
}

pub struct SessionBuilder {
    pub(crate) inner: Arc<Mutex<NamedEnv>>,

    pub(crate) name: String,
    pub(crate) options: Option<SessionOptions>,
    pub(crate) opt_level: GraphOptimizationLevel,
    pub(crate) num_threads: i16,
    pub(crate) model_filename: PathBuf,
    pub(crate) use_cuda: bool,
}

impl SessionBuilder {
    pub fn with_options(mut self, options: SessionOptions) -> SessionBuilder {
        self.options = Some(options);
        self
    }

    pub fn with_cuda(mut self, use_cuda: bool) -> SessionBuilder {
        unimplemented!()
        // self.use_cuda = use_cuda;
        // self
    }

    pub fn with_optimization_level(mut self, opt_level: GraphOptimizationLevel) -> SessionBuilder {
        self.opt_level = opt_level;
        self
    }

    pub fn with_number_threads(mut self, num_threads: i16) -> SessionBuilder {
        self.num_threads = num_threads;
        self
    }

    pub fn build(self) -> Result<Session> {
        let mut session_options_ptr: *mut sys::OrtSessionOptions = std::ptr::null_mut();
        let status = unsafe { (*g_ort()).CreateSessionOptions.unwrap()(&mut session_options_ptr) };
        status_to_result(status).map_err(OrtError::SessionOptions)?;
        assert_eq!(status, std::ptr::null_mut());
        assert_ne!(session_options_ptr, std::ptr::null_mut());

        match self.options {
            Some(_options) => unimplemented!(),
            None => {}
        }

        // We use a u16 in the builder to cover the 16-bits positive values of a i32.
        let num_threads = self.num_threads as i32;
        unsafe { (*g_ort()).SetIntraOpNumThreads.unwrap()(session_options_ptr, num_threads) };

        // Sets graph optimization level
        let opt_level = self.opt_level as u32;
        unsafe {
            (*g_ort()).SetSessionGraphOptimizationLevel.unwrap()(session_options_ptr, opt_level)
        };

        let env_ptr: *const sys::OrtEnv = *self.inner.lock().unwrap().env_ptr.0.get_mut();
        let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

        if !self.model_filename.exists() {
            return Err(OrtError::FileDoesNotExists {
                filename: self.model_filename.clone(),
            });
        }
        let model_filename = self.model_filename.clone();
        let model_path: CString =
            CString::new(
                self.model_filename
                    .to_str()
                    .ok_or_else(|| OrtError::NonUtf8Path {
                        path: model_filename,
                    })?,
            )?;

        let status = unsafe {
            (*g_ort()).CreateSession.unwrap()(
                env_ptr,
                model_path.as_ptr(),
                session_options_ptr,
                &mut session_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::Session)?;
        assert_eq!(status, std::ptr::null_mut());
        assert_ne!(session_ptr, std::ptr::null_mut());

        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        let status =
            unsafe { (*g_ort()).GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr) };
        status_to_result(status).map_err(OrtError::Allocator)?;
        assert_eq!(status, std::ptr::null_mut());
        assert_ne!(allocator_ptr, std::ptr::null_mut());

        Ok(Session {
            session_ptr,
            allocator_ptr,
        })
    }
}

impl Session {
    fn read_inputs_count(&self) -> Result<u64> {
        let mut num_input_nodes: u64 = 0;
        let status = unsafe {
            (*g_ort()).SessionGetInputCount.unwrap()(self.session_ptr, &mut num_input_nodes)
        };
        status_to_result(status).map_err(OrtError::Allocator)?;
        assert_eq!(status, std::ptr::null_mut());
        assert_ne!(num_input_nodes, 0);
        Ok(num_input_nodes)
    }

    fn read_input_name(&self, i: u64) -> Result<String> {
        let mut input_name_bytes: *mut i8 = std::ptr::null_mut();

        let status = unsafe {
            (*g_ort()).SessionGetInputName.unwrap()(
                self.session_ptr,
                i,
                self.allocator_ptr,
                &mut input_name_bytes,
            )
        };
        status_to_result(status).map_err(OrtError::InputName)?;
        assert_ne!(input_name_bytes, std::ptr::null_mut());

        // FIXME: Is it safe to keep ownership of the memory
        let input_name = char_p_to_string(input_name_bytes)?;

        Ok(input_name)
    }

    fn read_input(&self, i: u64) -> Result<Input> {
        let input_name = self.read_input_name(i)?;

        let mut typeinfo_ptr: *mut sys::OrtTypeInfo = std::ptr::null_mut();

        let status = unsafe {
            (*g_ort()).SessionGetInputTypeInfo.unwrap()(
                self.session_ptr,
                i as u64,
                &mut typeinfo_ptr,
            )
        };
        status_to_result(status).map_err(OrtError::InputName)?;
        assert_ne!(typeinfo_ptr, std::ptr::null_mut());

        let mut tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
        let status = unsafe {
            (*g_ort()).CastTypeInfoToTensorInfo.unwrap()(typeinfo_ptr, &mut tensor_info_ptr)
        };
        status_to_result(status).map_err(OrtError::InputName)?;
        assert_ne!(tensor_info_ptr, std::ptr::null_mut());

        let mut input_type_sys: sys::ONNXTensorElementDataType = 0;
        let status = unsafe {
            (*g_ort()).GetTensorElementType.unwrap()(tensor_info_ptr, &mut input_type_sys)
        };
        status_to_result(status).map_err(OrtError::InputName)?;
        assert_ne!(input_type_sys, 0);
        // This transmute should be safe since its value is read from GetTensorElementType which we must trust.
        let input_type: TensorElementDataType = unsafe { std::mem::transmute(input_type_sys) };

        // println!("Input {} : type={}", i, type_);

        // print input shapes/dims
        let mut num_dims = 0;
        let status =
            unsafe { (*g_ort()).GetDimensionsCount.unwrap()(tensor_info_ptr, &mut num_dims) };
        status_to_result(status).map_err(OrtError::InputName)?;
        assert_ne!(num_dims, 0);

        // println!("Input {} : num_dims={}", i, num_dims);
        let mut input_node_dims: Vec<i64> = vec![0; num_dims as usize];
        let status = unsafe {
            (*g_ort()).GetDimensions.unwrap()(
                tensor_info_ptr,
                input_node_dims.as_mut_ptr(), // FIXME: UB?
                num_dims,
            )
        };
        status_to_result(status).map_err(OrtError::InputName)?;

        // for j in 0..num_dims {
        //     println!("Input {} : dim {}={}", i, j, input_node_dims[j as usize]);
        // }

        unsafe { (*g_ort()).ReleaseTypeInfo.unwrap()(typeinfo_ptr) };

        Ok(Input {
            name: input_name,
            input_type: input_type,
            dimensions: input_node_dims.into_iter().map(|d| d as u32).collect(),
        })
    }

    pub fn read_inputs(&self) -> Result<Vec<Input>> {
        let num_input_nodes = self.read_inputs_count()?;

        (0..num_input_nodes)
            .map(|i| self.read_input(i))
            .collect::<Result<Vec<Input>>>()
    }
}

#[derive(Debug)]
pub struct Input {
    name: String,
    input_type: TensorElementDataType,
    dimensions: Vec<u32>,
}