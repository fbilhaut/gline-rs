use cap::Cap;
use std::alloc::*;
use memory_stats::*;
use humansize::{format_size, DECIMAL};


#[global_allocator] 
static ALLOCATOR: Cap<System> = Cap::new(System, usize::max_value());


pub fn print_memory_usage() {
    println!("Memory usage:");
    println!("- Allocated: {} bytes ({})", ALLOCATOR.allocated(), format_size(ALLOCATOR.allocated(), DECIMAL));
    if let Some(usage) = memory_stats() {
        println!("- Physical: {} bytes ({})", usage.physical_mem, format_size(usage.physical_mem, DECIMAL));
        println!("- Virtual: {} bytes ({})", usage.virtual_mem, format_size(usage.virtual_mem, DECIMAL));
    }    
}