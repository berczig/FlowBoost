import torch
from nicegui import ui, app, Client
from pathlib import Path
import os
import datetime
import time
import asyncio
import matplotlib.pyplot as plt
import copy # For deepcopying metadata for editing
from flow_boost.spheres_in_cube.data_load_save import FILE_STRUCTURE

# --- Application State & Configuration ---
APP_STATE = {
    "model": {"folder":"models", "files": [], "sort_by": "name", "selected_file_path": None},
    "data": {"folder": "datasets", "files": [], "sort_by": "name", "selected_file_path": None},
    "edit": {"folder": "./", "files": [], "sort_by": "name", "selected_file_path": None},
    "merge": {"folder": "datasets", "files": [], "sort_by": "name", "selected_file_paths": set(), "output_filename": "merged_data.pt"}
}
VIEW_CONFIG = {
    "model": {"filter_type": "model", "title": "Model Files"},
    "data": {"filter_type": "data", "title": "Data Files"},
    "edit": {"filter_type": None, "title": "Edit Metadata"},
    "merge": {"filter_type": "data", "title": "Merge Data Files"} # Filter for 'data' type
}
SELECTED_FILE_CONTEXT = {
    "view_key": None, "file_path": None, "full_metadata": {},
    "system_info": {}, "content_keys": [], "edit_buffer": {},
    "tensor_shape": None # <-- ADDED
}
REQUIRED_FIELDS = FILE_STRUCTURE

# --- Helper Functions ---
def format_timestamp(ts):
    if ts is None: return "N/A"
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def load_file_as_dict(file_path, **args): # Used by load_pt_file_info
    file_content = torch.load(file_path, **args)
    if not isinstance(file_content, dict):
        # If it's not a dict (e.g., just a tensor), wrap it.
        # This helps standardize what load_pt_file_info expects.
        return {"content": file_content} 
    return file_content

async def load_file_as_dict_async(file_path, **args):
    return await asyncio.to_thread(load_file_as_dict, file_path, **args)

def format_size(size_bytes):
    if size_bytes is None: return "N/A"
    if size_bytes == 0: return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = 0
    s_bytes = float(size_bytes)
    while s_bytes >= 1024 and i < len(size_name) -1 :
        s_bytes /= 1024.0
        i += 1
    return f"{s_bytes:.2f} {size_name[i]}"

async def load_pt_file_info(file_path: Path):
    try:
        content = await load_file_as_dict_async(file_path, map_location='cpu')
        metadata = content.get('metadata', {})
        file_type = metadata.get('type', 'unknown')
        stat_info = await asyncio.to_thread(os.stat, file_path)

        tensor_shape_info = None # Initialize
        if file_type == "data":
            tensor_data = content.get("tensor")
            if isinstance(tensor_data, torch.Tensor):
                tensor_shape_info = str(list(tensor_data.shape)) # Store as string representation of list
            elif "tensor" in REQUIRED_FIELDS.get("data", {}): # Check if tensor key is defined as required
                tensor_shape_info = "Tensor key ('tensor') missing or not a valid Tensor object."
            # If "tensor" is not in REQUIRED_FIELDS["data"] but file is type "data", shape_info remains None

        missing_fields = []
        metadata_preview = {"creation_time_ts": metadata.get('creation_time_ts', stat_info.st_ctime)}
        if file_type != 'unknown' and file_type in REQUIRED_FIELDS:
            req_meta_fields = REQUIRED_FIELDS[file_type].get("metadata", {})
            for req_field, req_field_info in req_meta_fields.items():
                if req_field not in metadata:
                    missing_fields.append(req_field)
                elif req_field_info.get("display"):
                    metadata_preview[req_field] = metadata[req_field]
        
        return {
            "path": file_path, "name": file_path.name, "type": file_type,
            "metadata_preview": metadata_preview, "full_metadata": metadata,
            "system_info": {"file_size_bytes": stat_info.st_size, "modified_time_ts": stat_info.st_mtime},
            "missing_metadata_fields": missing_fields, "content_keys": list(content.keys()),
            "tensor_shape": tensor_shape_info, # <-- MODIFIED/ADDED
        }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return { 
            "path": file_path, "name": file_path.name, "type": "error", 
            "full_metadata": {"error_message": str(e)}, "system_info": {},
            "missing_metadata_fields": [], "error_message": str(e), "content_keys": [],
            "tensor_shape": None # <-- MODIFIED/ADDED
        }

async def scan_folder_for_view(view_key: str):
    state = APP_STATE[view_key]
    config = VIEW_CONFIG[view_key]
    filter_type = config["filter_type"]

    if not state["folder"]:
        state["files"] = []
        get_file_list_container(view_key).refresh()
        return

    folder_path = Path(state["folder"])
    found_files_info = []
    if folder_path.is_dir():
        pt_files = list(folder_path.rglob("*.pt"))
        tasks = [load_pt_file_info(f) for f in pt_files]
        results = await asyncio.gather(*tasks)
        
        for info in results:
            if info:
                if filter_type is None or info["type"] == filter_type or info["type"] == "error":
                    # Error files are shown in Edit, or if they were meant to be data/model but failed to load type
                    if filter_type is not None and info["type"] == "error" and info["name"].endswith(".pt"):
                        # If we are filtering for a type, error files are generally not of that type
                        # but for Edit mode (filter_type is None) they are fine.
                        # For specific types, only include if type matches (or error during load of typed file)
                         pass # Allow error files to be shown if they are .pt
                    found_files_info.append(info)

    sort_key_map = {
        "name": lambda x: x["name"].lower(),
        "date": lambda x: x.get("metadata_preview", {}).get("creation_time_ts", x.get("system_info",{}).get("modified_time_ts", 0)) or 0,
        "size": lambda x: x.get("system_info",{}).get("file_size_bytes", 0) or 0,
    }
    found_files_info.sort(key=sort_key_map[state["sort_by"]], reverse=(state["sort_by"] in ["date", "size"]))
    state["files"] = found_files_info

    # Clear selection if selected file is no longer present
    if view_key == "merge":
        current_selection = state["selected_file_paths"]
        valid_paths_in_new_scan = {str(f["path"]) for f in found_files_info}
        state["selected_file_paths"] = current_selection.intersection(valid_paths_in_new_scan)
        if not state["selected_file_paths"]: # If all selected files are gone
             clear_selected_file_context_and_refresh_details() # Ensures right panel updates if it was showing merge controls
        else: # Some selection remains, refresh panels
            details_panel_content_area.refresh()

    elif state["selected_file_path"] and not any(f["path"] == state["selected_file_path"] for f in found_files_info):
        state["selected_file_path"] = None
        clear_selected_file_context_and_refresh_details()

    get_file_list_container(view_key).refresh()
    if (view_key != "merge" and not state["selected_file_path"]) or \
       (view_key == "merge" and not state["selected_file_paths"]):
         clear_selected_file_context_and_refresh_details()


def get_file_list_container(view_key: str):
    if view_key == "model": return model_files_container
    if view_key == "data": return data_files_container
    if view_key == "edit": return edit_files_container
    if view_key == "merge": return merge_files_container
    return None

def clear_selected_file_context_and_refresh_details():
    SELECTED_FILE_CONTEXT.update({
        "view_key": None, "file_path": None, "full_metadata": {}, 
        "system_info": {}, "content_keys": [], "edit_buffer": {},
        "tensor_shape": None # <-- ADDED
    })
    # If the active tab was "merge", its specific selections in APP_STATE are handled by scan_folder_for_view
    details_panel_content_area.refresh()

async def handle_folder_select_for_view(view_key: str, e):
    APP_STATE[view_key]["folder"] = e.path
    if view_key == "merge":
        APP_STATE[view_key]["selected_file_paths"].clear()
    else:
        APP_STATE[view_key]["selected_file_path"] = None
    clear_selected_file_context_and_refresh_details()
    await scan_folder_for_view(view_key)

# --- File Conversion Logic ---
class ConversionOptionsDialog(ui.dialog):
    def __init__(self, original_file_path: Path, target_type: str, on_confirm_callback):
        super().__init__()
        self.original_file_path = original_file_path
        self.target_type = target_type
        self.on_confirm_callback = on_confirm_callback

        with self, ui.card().classes('min-w-[450px]'):
            ui.label(f"Convert '{original_file_path.name}' to '{target_type.capitalize()}' type?").classes("text-lg")
            
            self.delete_original_switch = ui.switch("Delete original file after conversion?", value=False)
            
            default_new_name = f"{original_file_path.stem}_as_{target_type}{original_file_path.suffix}"
            self.new_name_input = ui.input("New file name (if not deleting original)", value=default_new_name) \
                .props('outlined dense').classes('w-full')
            
            self.new_name_input.bind_visibility_from(self.delete_original_switch, 'value', backward=lambda x: not x)

            with ui.row().classes("w-full justify-end mt-4 gap-2"):
                ui.button("Cancel", on_click=self.close).props("flat")
                ui.button("Confirm Conversion", on_click=self._handle_confirm).props("color=primary")

    async def _handle_confirm(self):
        delete_original = self.delete_original_switch.value
        new_name = self.new_name_input.value if not delete_original else None
        self.close()
        await self.on_confirm_callback(
            original_file_path_str=str(self.original_file_path),
            target_type=self.target_type,
            delete_original=delete_original,
            new_file_name_str=new_name
        )

async def process_file_conversion(original_file_path_str: str, target_type: str, delete_original: bool, new_file_name_str: str = None):
    original_file_path = Path(original_file_path_str)
    parent_folder = original_file_path.parent
    
    if delete_original:
        final_save_path = original_file_path
    else:
        if not new_file_name_str:
            new_file_name_str = f"{original_file_path.stem}_as_{target_type}{original_file_path.suffix}"
        final_save_path = parent_folder / new_file_name_str
        if final_save_path.exists() and final_save_path == original_file_path: # Safety for renaming to same
             final_save_path = parent_folder / f"{original_file_path.stem}_as_{target_type}_{int(time.time())}{original_file_path.suffix}"


    try:
        raw_original_content = await asyncio.to_thread(torch.load, original_file_path, map_location='cpu')
        new_pt_structure = {}
        
        stat_info = await asyncio.to_thread(os.stat, original_file_path)
        new_metadata = {"type": target_type, "creation_time_ts": stat_info.st_ctime}
        if target_type in REQUIRED_FIELDS:
            for req_meta_key in REQUIRED_FIELDS[target_type].get("metadata", {}):
                if req_meta_key not in new_metadata:
                    new_metadata[req_meta_key] = "" # Placeholder for required fields
        new_pt_structure["metadata"] = new_metadata

        if target_type == "data":
            if "tensor" not in REQUIRED_FIELDS["data"]: # Should not happen with current config
                ui.notify("Internal config error: 'tensor' key not defined for data type.", type='error'); return

            extracted_tensor = None
            if isinstance(raw_original_content, torch.Tensor):
                extracted_tensor = raw_original_content
            elif isinstance(raw_original_content, dict):
                # Try to find a tensor in common keys
                for key_try in ['tensor', 'content', 'data']: 
                    if isinstance(raw_original_content.get(key_try), torch.Tensor):
                        extracted_tensor = raw_original_content[key_try]
                        # Copy other potentially useful keys from original dict, avoiding reserved ones
                        for k, v in raw_original_content.items():
                            if k not in [key_try, 'metadata'] and k not in new_pt_structure:
                                new_pt_structure[k] = v
                        break
                if extracted_tensor is None:
                    ui.notify(f"Could not find a primary tensor in '{original_file_path.name}' for 'data' conversion. Original keys: {list(raw_original_content.keys())}. Saving dict as is under 'original_content'.", type='warning')
                    new_pt_structure['original_content'] = raw_original_content # Store the whole dict
            
            if extracted_tensor is not None:
                new_pt_structure["tensor"] = extracted_tensor
            elif 'original_content' not in new_pt_structure : # If no tensor found and not stored as original_content
                 ui.notify(f"Original file type ({type(raw_original_content)}) not directly usable as tensor for 'data' type. Storing as 'payload'.", type='warning')
                 new_pt_structure['payload'] = raw_original_content


        elif target_type == "model":
            if isinstance(raw_original_content, dict):
                for key, value in raw_original_content.items():
                    if key != "metadata": # Preserve all but old metadata
                        new_pt_structure[key] = value
            else: # E.g. raw_original_content is a state_dict or single tensor
                new_pt_structure["model_content"] = raw_original_content
        
        await asyncio.to_thread(torch.save, new_pt_structure, final_save_path)
        ui.notify(f"File converted to '{target_type}' and saved as '{final_save_path.name}'.", type='positive')

        if delete_original and original_file_path.exists() and final_save_path != original_file_path:
            await asyncio.to_thread(os.remove, original_file_path)
            ui.notify(f"Original file '{original_file_path.name}' deleted.", type='info')
        
        # Refresh current view ('edit' in this context) and clear details
        active_view = SELECTED_FILE_CONTEXT.get("view_key", "edit") # Fallback to edit
        await scan_folder_for_view(active_view)
        clear_selected_file_context_and_refresh_details()

    except Exception as e:
        print(f"Error during conversion of {original_file_path_str}: {e}")
        ui.notify(f"Conversion error: {e}", type='negative', multi_line=True)


# --- File Card and Selection ---
def create_file_card_for_view(file_info, view_key: str):
    state = APP_STATE[view_key]
    is_selected = False
    if view_key == "merge":
        is_selected = str(file_info["path"]) in state.get("selected_file_paths", set())
    else:
        is_selected = state.get("selected_file_path") == file_info["path"]
    
    card_action = lambda fi=file_info, vk=view_key: (
        handle_merge_file_toggle_select(fi) if vk == "merge" 
        else handle_file_select(fi, vk)
    )

    with ui.card().tight().classes('w-full cursor-pointer hover:shadow-lg transition-shadow relative') \
            .on('click', card_action):
        card_classes = 'w-full'
        if file_info["type"] == "error": card_classes += ' bg-red-100 dark:bg-red-900'
        
        with ui.card_section().classes(card_classes):
            ui.label(file_info["name"]).classes('font-bold text-sm')
            if file_info["type"] == "error":
                ui.label(f"Error: {file_info.get('error_message', 'Failed to load')}").classes('text-xs text-red-600 break-all')
            else:
                meta_preview = file_info.get("metadata_preview", {})
                actual_type = file_info.get("type", "unknown")
                ui.label(f"Type: {actual_type}").classes('text-xs font-semibold')
                created_str = format_timestamp(meta_preview.get("creation_time_ts"))
                ui.label(f"Created: {created_str}").classes('text-xs')
                size_str = format_size(file_info.get("system_info",{}).get("file_size_bytes"))
                ui.label(f"File Size: {size_str}").classes('text-xs')

                if actual_type != 'unknown' and actual_type in REQUIRED_FIELDS:
                    for field, info in REQUIRED_FIELDS[actual_type].get("metadata", {}).items():
                        if info.get("display"):
                            if field in meta_preview:
                                ui.label(f"{info['display_label']}: {meta_preview[field]}").classes('text-xs')
                            # else: No need to warn here, missing_metadata_fields icon handles it

        if file_info.get("missing_metadata_fields"):
            with ui.button(icon='warning', color='orange').props('flat dense round absolute top-1 right-1 z-10'):
                missing_list = "\n".join(file_info["missing_metadata_fields"])
                ui.tooltip(f"Missing metadata:\n{missing_list}").classes('bg-orange-100 text-black whitespace-pre-line')
        
        if is_selected:
            if view_key == "merge": # Different highlight for multi-select
                ui.html().classes('absolute top-0 left-0 right-0 bottom-0 border-4 border-accent rounded-md pointer-events-none')
            else: # Single select highlight
                ui.html().classes('absolute top-0 right-0 bottom-0 w-1 bg-primary pointer-events-none')

def handle_file_select(file_info, view_key: str): # For single selection views
    state = APP_STATE[view_key]
    if file_info["type"] == "error":
        ui.notify(f"Cannot select file with load error: {file_info['name']}", type='negative')
        if state["selected_file_path"] == file_info["path"]:
            state["selected_file_path"] = None
            clear_selected_file_context_and_refresh_details()
            get_file_list_container(view_key).refresh()
        return

    state["selected_file_path"] = file_info["path"]
    SELECTED_FILE_CONTEXT.update({
        "view_key": view_key, "file_path": file_info["path"],
        "full_metadata": copy.deepcopy(file_info["full_metadata"]),
        "system_info": file_info["system_info"], "content_keys": file_info["content_keys"],
        "edit_buffer": copy.deepcopy(file_info["full_metadata"]) if view_key == "edit" else {},
        "tensor_shape": file_info.get("tensor_shape") # <-- ADDED
    })
    details_panel_content_area.refresh()
    get_file_list_container(view_key).refresh()

def handle_merge_file_toggle_select(file_info): # For multi-selection in Merge view
    view_key = "merge"
    state = APP_STATE[view_key]
    if file_info["type"] == "error":
        ui.notify(f"Cannot select file with load error for merge: {file_info['name']}", type='negative'); return
    if file_info["type"] != "data": # Should be pre-filtered, but double check
        ui.notify(f"Only 'data' type files can be merged. '{file_info['name']}' is '{file_info['type']}'.", type='warning'); return

    file_path_str = str(file_info["path"])
    if file_path_str in state["selected_file_paths"]:
        state["selected_file_paths"].remove(file_path_str)
    else:
        state["selected_file_paths"].add(file_path_str)

    SELECTED_FILE_CONTEXT.update({"view_key": view_key, "file_path": None}) # Clear single file context
    details_panel_content_area.refresh() # Update right panel to show merge controls
    get_file_list_container(view_key).refresh() # Update card highlights

# --- UI: File List & Main Content for Views ---
@ui.refreshable
def view_specific_file_list(view_key: str):
    state = APP_STATE[view_key]
    if not state["folder"]:
        ui.label("Select a folder.").classes('m-4 text-center text-gray-500')
        return
    if not state["files"]:
        filter_msg = f" for type '{VIEW_CONFIG[view_key]['filter_type']}'" if VIEW_CONFIG[view_key]['filter_type'] else ""
        ui.label(f"No .pt files found{filter_msg} in this folder.").classes('m-4 text-center text-gray-500')
    
    if state["files"]: ui.label(f"Loaded {len(state['files'])} files").classes("font-medium text-green-600 mt-2")
    with ui.grid(columns=3).classes('gap-4 w-full'):
        for file_info in state["files"]:
            create_file_card_for_view(file_info, view_key)

def create_main_list_content_for_view(view_key: str):
    state = APP_STATE[view_key]
    config = VIEW_CONFIG[view_key]

    with ui.column().classes('w-full p-3 gap-3'):
        ui.label(config["title"]).classes("text-lg font-semibold")
        
        async def dialog_callback_for_folder_select(e):
            await handle_folder_select_for_view(view_key, e)

        def open_folder_input_dialog():
            dialog = FolderInputDialog(
                current_path=state.get("folder"),
                on_select_callback=dialog_callback_for_folder_select,
                title=f"Select Folder for {config['title']}"
            )
            dialog.open()
        
        with ui.row():
            ui.button('Select Folder', icon='folder_open', on_click=open_folder_input_dialog).props('outline rounded')
            
            # Define the label that shows the truncated folder name
            folder_display_label = ui.label().bind_text_from(
                state, "folder",
                backward=lambda x: f"Selected Folder: \'{Path(x).name if x else 'N/A'}\'" if x else "No folder selected."
            ).classes('font-medium text-blue-1000 mt-2')
        
        # Add a tooltip to the label, showing the full folder path if available.
        # The tooltip is made a child of the label for explicit association.
        # Its visibility is bound to whether state["folder"] has a value.
        with folder_display_label:
            ui.tooltip().bind_text_from(state, "folder") \
                        .bind_visibility_from(state, "folder", backward=bool)

        with ui.row().classes('items-center w-full'):
            ui.select(options={"name": "Name", "date": "Date", "size": "File Size"}, label="Sort by", 
                      value=state["sort_by"],
                      on_change=lambda e, vk=view_key: (APP_STATE[vk].__setitem__("sort_by", e.value), 
                                                        asyncio.create_task(scan_folder_for_view(vk)))
            ).classes('flex-grow text-sm').props('dense outlined')
            ui.button(icon='refresh', on_click=lambda vk=view_key: scan_folder_for_view(vk)) \
                .props('flat round dense').tooltip("Refresh file list")
        
        ui.separator()
        scroll_height = 'h-[calc(100vh-250px)]'
        with ui.scroll_area().classes(f'w-full {scroll_height} border rounded-md p-1'):
            container = get_file_list_container(view_key)
            if container: container()

# --- Merge Operation Logic ---
async def perform_merge_operation():
    merge_state = APP_STATE["merge"]
    selected_paths_str = list(merge_state["selected_file_paths"])
    if len(selected_paths_str) < 2:
        ui.notify("Please select at least two 'data' files to merge.", type='warning'); return

    output_filename = merge_state.get("output_filename", f"merged_data_{int(time.time())}.pt")
    if not output_filename.endswith(".pt"): output_filename += ".pt"
    
    output_path = Path(merge_state["folder"]) / output_filename
    if output_path.exists():
        # TODO: Could add a confirmation dialog for overwrite
        ui.notify(f"Output file {output_filename} already exists. Merging aborted. Choose a new name.", type='error'); return

    loaded_contents = []
    first_file_meta = None
    ref_shape_suffix = None
    total_packings = 0

    try:
        for p_str in selected_paths_str:
            path = Path(p_str)
            # Use load_file_as_dict_async as it gives raw dict with metadata and tensor keys
            content = await load_file_as_dict_async(path, map_location='cpu') # This loads the file, NOT load_pt_file_info
            
            if not isinstance(content, dict) or "tensor" not in content or not isinstance(content["tensor"], torch.Tensor):
                ui.notify(f"File '{path.name}' is not in expected format (missing 'tensor'). Aborting.", type='error'); return
            if content.get("metadata", {}).get("type") != "data":
                 ui.notify(f"File '{path.name}' is not of type 'data'. Aborting.", type='error'); return

            tensor = content["tensor"]
            if tensor.ndim < 1: # Must have at least one dimension for concatenation
                 ui.notify(f"Tensor in '{path.name}' has no dimensions. Aborting.", type='error'); return

            current_shape_suffix = tensor.shape[1:] if tensor.ndim > 1 else tuple()
            if ref_shape_suffix is None:
                ref_shape_suffix = current_shape_suffix
                first_file_meta = content.get("metadata", {})
            elif current_shape_suffix != ref_shape_suffix:
                ui.notify(f"Tensor dimension mismatch. Expected trailing shape {ref_shape_suffix}, "
                          f"but '{path.name}' has {current_shape_suffix}. Aborting.", type='error'); return
            
            loaded_contents.append(content)

            packings_in_current_file = 0
            n_packings_from_metadata = content.get("metadata", {}).get("n_packings")
            if isinstance(n_packings_from_metadata, int): packings_in_current_file = n_packings_from_metadata
            elif isinstance(n_packings_from_metadata, str) and n_packings_from_metadata.isdigit(): packings_in_current_file = int(n_packings_from_metadata)
            else:packings_in_current_file = tensor.shape[0] if tensor.ndim > 0 else 0 # Fallback to tensor.shape[0] if metadata value is missing,
            
            total_packings += packings_in_current_file

        tensors_to_merge = [lc["tensor"] for lc in loaded_contents]
        merged_tensor = torch.cat(tensors_to_merge, dim=0)

        new_metadata = {
            "type": "data",
            "creation_time_ts": time.time(),
            "n_packings": total_packings,
            "n_spheres_per_packing": first_file_meta.get("n_spheres_per_packing", ""), # Take from first, or default
            "dimension": first_file_meta.get("dimension", ""), # Take from first, or default
            "source_files": [Path(p).name for p in selected_paths_str],
            "merged_from_n_files": len(selected_paths_str)
        }
        # Add any other required data fields with defaults if not covered
        for req_key in REQUIRED_FIELDS["data"].get("metadata", {}):
            if req_key not in new_metadata:
                new_metadata[req_key] = ""


        new_pt_content = {"metadata": new_metadata, "tensor": merged_tensor}
        await asyncio.to_thread(torch.save, new_pt_content, output_path)
        ui.notify(f"Successfully merged {len(tensors_to_merge)} files into '{output_path.name}'.", type='positive')

        merge_state["selected_file_paths"].clear()
        await scan_folder_for_view("merge") # Refresh file list
        details_panel_content_area.refresh() # Refresh right panel

    except Exception as e:
        print(f"Error during merge operation: {e}")
        ui.notify(f"Merge error: {str(e)}", type='negative', multi_line=True)

# --- UI: Right Details Panel ---
@ui.refreshable
def render_details_panel_content():
    ctx = SELECTED_FILE_CONTEXT
    
    if ctx.get("view_key") == "merge": # Special panel for Merge view
        with ui.card().classes('w-full h-full overflow-hidden'):
            with ui.card_section().classes('bg-gray-100 dark:bg-gray-700'):
                ui.label("Merge Controls").classes("text-lg font-semibold")
            with ui.scroll_area().classes('h-[calc(100%-70px)]'):
                 with ui.card_section():
                    merge_state = APP_STATE["merge"]
                    selected_for_merge = merge_state["selected_file_paths"]
                    if not selected_for_merge:
                        ui.label("Select two or more 'data' files from the list to merge.")
                    else:
                        ui.label(f"{len(selected_for_merge)} files selected for merging:").classes("font-medium")
                        with ui.list().props('dense bordered separator rounded-borders').classes("my-2 bg-white dark:bg-gray-800"):
                            for p_str in sorted(list(selected_for_merge)): # Sorted for consistent display
                                ui.item(Path(p_str).name).classes("text-xs")
                        
                        ui.input("Output file name for merged data",
                                value=merge_state["output_filename"],
                                on_change=lambda e: merge_state.__setitem__("output_filename", e.value)) \
                            .props('dense outlined clearable').classes("w-full mt-2")
                        
                        ui.button("Merge Dataset", on_click=perform_merge_operation) \
                            .props(f"color=positive rounded {'disable' if len(selected_for_merge) < 2 else ''}").classes("mt-4 w-full")
        return # End of merge specific rendering

    # --- Standard Details Panel for Model/Data/Edit ---
    if not ctx["file_path"]:
        ui.label("Select a file to see its details.").classes("m-4 text-gray-500 text-center")
        return

    file_name = Path(ctx["file_path"]).name
    with ui.card().classes('w-full h-full overflow-hidden'):
        with ui.card_section().classes('bg-gray-100 dark:bg-gray-700'):
            ui.label(f"Details: {file_name}").classes("text-lg font-semibold")
            ui.label(f"View: {ctx['view_key'].capitalize()}").classes("text-xs text-gray-500")
        
        # Conversion buttons for 'Edit' view and 'unknown' type files
        if ctx["view_key"] == "edit" and \
           (not ctx["full_metadata"] or ctx["full_metadata"].get("type", "unknown") == "unknown"):
            
            def open_conversion_dialog_wrapper(target_type):
                dialog = ConversionOptionsDialog(
                    original_file_path=Path(ctx["file_path"]),
                    target_type=target_type,
                    on_confirm_callback=process_file_conversion
                )
                dialog.open()

            with ui.row().classes('justify-center my-2 gap-2 p-2 border-t border-b'):
                ui.button("Convert to Data Type", on_click=lambda: open_conversion_dialog_wrapper("data")).props("color=blue outline dense")
                ui.button("Convert to Model Type", on_click=lambda: open_conversion_dialog_wrapper("model")).props("color=purple outline dense")
        
        ui.separator().classes('mb-2') # Separator after header/conversion

        with ui.scroll_area().classes('h-[calc(100%-70px)]'): # Adjust height if conversion section is visible
            with ui.card_section():
                # Display System Info
                ui.label("System Information:").classes("font-medium text-blue-600 mt-2")
                if ctx["system_info"]:
                    ui.label(f"Full Path: {ctx['file_path']}").classes("text-xs ml-2 break-all")
                    f_size = format_size(ctx["system_info"].get('file_size_bytes'))
                    ui.label(f"File Size: {f_size}").classes("text-xs ml-2")
                    mod_time = format_timestamp(ctx["system_info"].get('modified_time_ts'))
                    ui.label(f"Last Modified: {mod_time}").classes("text-xs ml-2")
                ui.label(f"Top-level keys in .pt: {ctx['content_keys']}").classes("text-xs ml-2 text-gray-500")
                
                # --- Display Tensor Shape for 'data' view ---
                if ctx["view_key"] == "data":
                    tensor_shape_value = ctx.get("tensor_shape")
                    if tensor_shape_value: # If shape info is available (not None or empty)
                        ui.label("Tensor Information:").classes("font-medium text-purple-600 mt-3")
                        ui.label(f"Shape: {tensor_shape_value}").classes("text-sm ml-2")
                    elif "tensor" in REQUIRED_FIELDS.get("data", {}): # If tensor is required but shape wasn't found
                        ui.label("Tensor Information:").classes("font-medium text-purple-600 mt-3")
                        ui.label("Shape: Not available (tensor might be missing or invalid).").classes("text-sm ml-2 text-orange-600")
                # --- End Display Tensor Shape ---

                if 'metadata' not in ctx['content_keys'] and ctx["full_metadata"]:
                    ui.badge("Warning: 'metadata' key not top-level, but metadata found.", color='orange').classes('text-xs')
                ui.separator().classes('my-2')

                target_metadata_dict = ctx["edit_buffer"] if ctx["view_key"] == "edit" else ctx["full_metadata"]
                if not target_metadata_dict and ctx["full_metadata"].get("error_message"):
                     ui.label("Error loading metadata:").classes("font-medium text-red-600 mt-2")
                     ui.label(ctx["full_metadata"]["error_message"]).classes("text-sm ml-2 text-red-500")
                elif not target_metadata_dict:
                     ui.label("No metadata found or file is of 'unknown' type.").classes("text-sm ml-2 text-gray-500")
                     if ctx["view_key"] != "edit": ui.label("Switch to 'Edit' view to assign a type or add metadata.").classes("text-xs ml-2 text-gray-400")
                else:
                    ui.label("Metadata:").classes("font-medium text-green-600 mt-2")
                    render_metadata_items(target_metadata_dict, ctx["view_key"] == "edit")

                if ctx["view_key"] == "edit":
                    render_edit_view_controls()


def render_metadata_items(metadata_dict, is_editable: bool):
    for key, value in list(metadata_dict.items()):
        with ui.row().classes('items-center w-full hover:bg-gray-50 dark:hover:bg-gray-800 py-1'):
            key_display = key.replace('_', ' ').title()
            if is_editable:
                if isinstance(value, str):
                    ui.input(label=key_display, value=value, 
                             on_change=lambda e, k=key: SELECTED_FILE_CONTEXT["edit_buffer"].__setitem__(k, e.value)) \
                        .props('dense outlined').classes('flex-grow')
                elif isinstance(value, (int, float)):
                    ui.number(label=key_display, value=value, 
                              on_change=lambda e, k=key: SELECTED_FILE_CONTEXT["edit_buffer"].__setitem__(k, e.value), 
                              format='%.6g' if isinstance(value, float) else None) \
                        .props('dense outlined').classes('flex-grow')
                elif isinstance(value, bool):
                    ui.switch(text=key_display, value=value, 
                              on_change=lambda e, k=key: SELECTED_FILE_CONTEXT["edit_buffer"].__setitem__(k, e.value))
                else: render_readonly_metadata_item(key_display, value) # Complex types readonly in edit
                
                if key not in ['creation_time_ts']: # ['type', 'creation_time_ts'] Protect some core fields
                    ui.button(icon='delete_outline', color='negative',
                              on_click=lambda k=key: (SELECTED_FILE_CONTEXT["edit_buffer"].pop(k, None), details_panel_content_area.refresh())) \
                        .props('flat round dense').tooltip(f"Delete '{key}'")
            else:
                render_readonly_metadata_item(key_display, value)

def render_readonly_metadata_item(display_key, value):
    ui.label(f"{display_key}:").classes('font-medium text-sm mr-2 min-w-[120px]')
    if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value) \
       and any(kw in display_key.lower() for kw in ['history', 'loss', 'accuracy', 'metric']):
        with ui.element('div').classes('w-full my-1'):
            with ui.pyplot(figsize=(4.5, 2.5), close=True):
                plt.plot(value); plt.title(display_key, fontsize=10)
                plt.xlabel("Step", fontsize=8); plt.ylabel("Value", fontsize=8)
                plt.xticks(fontsize=7); plt.yticks(fontsize=7)
                plt.grid(True); plt.tight_layout(pad=0.5)
    elif isinstance(value, torch.Tensor):
        ui.label(f"Tensor (shape: {value.shape}, dtype: {value.dtype})").classes('text-sm break-all')
    elif isinstance(value, dict):
        with ui.expansion(display_key, icon='schema').classes('w-full text-sm'):
            with ui.card().tight():
                with ui.card_section():
                    for sub_key, sub_val in value.items():
                        render_readonly_metadata_item(sub_key.replace('_', ' ').title(), sub_val)
    elif isinstance(value, list):
        ui.label(f"{len(value)} items (first 5 shown):").classes('text-xs text-gray-500')
        for item_idx, item in enumerate(value[:5]):
            ui.label(f"- {str(item)[:80]}{'...' if len(str(item)) > 80 else ''}").classes('text-sm ml-2')
        if len(value) > 5: ui.label("...").classes('text-xs ml-2 text-gray-500')
    else:
        ui.label(str(value)).classes('text-sm break-all')


def render_edit_view_controls():
    ui.separator().classes('my-3')
    with ui.expansion("Add New Metadata Item", icon='add_circle_outline').classes('w-full'):
        with ui.card().tight():
            with ui.card_section():
                new_item_state = {"key": "", "type": "str", "value_str": "", "value_num": 0, "value_bool": False}

                new_key_input = ui.input("Key Name", value=new_item_state["key"],
                                         on_change=lambda e: new_item_state.__setitem__("key", e.value)
                                         ).props('dense outlined')
                
                value_container = ui.column().classes('w-full mt-1')

                def update_value_input():
                    value_container.clear()
                    with value_container:
                        val_type = new_item_state["type"]
                        if val_type == "str":
                            ui.input("Value (String)", value=new_item_state["value_str"],
                                     on_change=lambda e: new_item_state.__setitem__("value_str", e.value)
                                     ).props('dense outlined').classes('flex-grow')
                        elif val_type in ["int", "float"]:
                            ui.number("Value (Number)", value=new_item_state["value_num"],
                                      on_change=lambda e: new_item_state.__setitem__("value_num", e.value),
                                      format='%.6g' if val_type == "float" else None
                                      ).props('dense outlined').classes('flex-grow')
                        elif val_type == "bool":
                            ui.switch("Value (Boolean)", value=new_item_state["value_bool"],
                                      on_change=lambda e: new_item_state.__setitem__("value_bool", e.value))
                
                ui.select({"str": "String", "int": "Integer", "float": "Float", "bool": "Boolean"}, 
                          value=new_item_state["type"], label="Type", 
                          on_change=lambda e: (new_item_state.__setitem__("type", e.value), update_value_input())
                          ).props('dense outlined')
                
                update_value_input() # Initial call

                def add_new_item():
                    key = new_item_state["key"].strip()
                    if not key: ui.notify("Key cannot be empty.", type='warning'); return
                    if key in SELECTED_FILE_CONTEXT["edit_buffer"]:
                        ui.notify(f"Key '{key}' already exists.", type='warning'); return
                    
                    val_type = new_item_state["type"]
                    value_to_add = None
                    try:
                        if val_type == "str": value_to_add = new_item_state["value_str"]
                        elif val_type == "int": value_to_add = int(new_item_state["value_num"])
                        elif val_type == "float": value_to_add = float(new_item_state["value_num"])
                        elif val_type == "bool": value_to_add = new_item_state["value_bool"]
                    except ValueError: ui.notify("Invalid number format.", type='negative'); return

                    SELECTED_FILE_CONTEXT["edit_buffer"][key] = value_to_add
                    new_item_state.update({"key": "", "value_str": "", "value_num": 0, "value_bool": False}) # Reset
                    new_key_input.set_value("") # Clear input
                    update_value_input() # Refresh value input to reflect reset state (might not be strictly needed visually)
                    details_panel_content_area.refresh()
                    ui.notify(f"Added '{key}'. Save changes to persist.", type='positive')

                ui.button("Add Item", on_click=add_new_item).props("color=primary dense").classes("mt-2")

    ui.button("Save Changes to File", icon='save', color='positive', 
              on_click=save_edited_metadata_to_file).props('rounded mt-4 w-full')


async def save_edited_metadata_to_file():
    ctx = SELECTED_FILE_CONTEXT
    if not ctx["file_path"] or ctx["view_key"] != "edit":
        ui.notify("Cannot save: No file selected in Edit view.", type='error'); return
    
    file_to_save = Path(ctx["file_path"])
    try:
        # Load original content, preserving non-metadata parts
        original_full_content = await asyncio.to_thread(torch.load, file_to_save, map_location='cpu')
        if not isinstance(original_full_content, dict): # Should be a dict after conversion or if it was one
            # This case means we might be editing a file that load_file_as_dict wrapped.
            # We want to save it back as a proper dictionary with 'metadata' and other keys.
            temp_content = original_full_content # preserve original if not dict
            original_full_content = {}
            # Decide where to put temp_content based on its nature or if it was 'content' from earlier load
            # This part can be tricky if the file was just a raw tensor initially.
            # Assuming if we're here, it was wrapped, or the user is building structure.
            # For simplicity, if it's not a dict, the 'metadata' will be added,
            # and the original non-dict content might be under 'content' key or similar
            # if that's how it was loaded by load_pt_file_info's helper.
            # A safer bet: the file *must* be a dict to save metadata *into* it.
            # If load_pt_file_info wrapped it, it would be {'content':...}
            # The conversion process should ensure it's a dict.
            # If still not a dict, it's an edge case, maybe it was created outside.
            # For now, let's assume it *will* be a dict or conversion made it so.
            ui.notify("Trying to save metadata to a file that is not a dictionary. This is unexpected for .pt files with metadata.", type="warning")
            # If it was loaded as {'content': tensor}, this is fine.
            # If it was loaded as just a tensor, torch.load would give tensor.
            # process_file_conversion makes it a dict.
            # So, it SHOULD be a dict here.
            
        original_full_content['metadata'] = copy.deepcopy(ctx["edit_buffer"])
        
        await asyncio.to_thread(torch.save, original_full_content, file_to_save)
        ui.notify(f"Metadata saved for {file_to_save.name}", type='positive')

        updated_info = await load_pt_file_info(file_to_save)
        if updated_info:
            view_state = APP_STATE[ctx["view_key"]]
            for i, f_info in enumerate(view_state['files']):
                if f_info['path'] == file_to_save:
                    view_state['files'][i] = updated_info; break
            handle_file_select(updated_info, ctx["view_key"]) # Reselect to refresh all
        else: await scan_folder_for_view(ctx["view_key"])

    except Exception as e:
        print(f"Error saving metadata for {file_to_save}: {e}")
        ui.notify(f"Error saving: {e}", type='negative', multi_line=True)

# --- Custom Folder Input Dialog & Event ---
class SelectedEvent: # Simple event object for path
    def __init__(self, path_obj: Path): self.path = str(path_obj)

class FolderInputDialog(ui.dialog):
    def __init__(self, current_path: str = None, on_select_callback=None, title: str = "Select Folder"):
        super().__init__()
        self.on_select_callback = on_select_callback
        with self, ui.card().classes('min-w-[400px]'):
            ui.label(title).classes("text-lg font-semibold")
            self.path_input = ui.input(label="Folder Path on Server", value=str(current_path or "")) \
                .classes("w-full").props('autofocus clearable')
            with ui.row().classes("w-full justify-end mt-4 gap-2"):
                ui.button("Cancel", on_click=self.close).props("flat")
                ui.button("Select", on_click=self._handle_select).props("color=primary")

    async def _handle_select(self):
        selected_path = Path(self.path_input.value.strip())
        if not selected_path: ui.notify("Path cannot be empty.", type='warning'); return
        if selected_path.is_dir():
            if self.on_select_callback: await self.on_select_callback(SelectedEvent(selected_path))
            self.close()
        else: ui.notify(f"Path not found or not a directory: {selected_path}", type='negative')

# --- Page Layout ---
@ui.page('/')
async def main_page(client: Client):
    global model_files_container, data_files_container, edit_files_container, merge_files_container
    global details_panel_content_area

    ui.query('body').style('padding: 0 !important; margin: 0; overflow: hidden;')
    with ui.header(elevated=True).classes('bg-primary text-white items-center justify-between h-[50px]'):
        ui.label("Sphere Packing - Result Viewer (Refresh page if nothing happens)").classes('text-xl font-bold')
        dark = ui.dark_mode()
        with ui.row(): # Keep dark/light buttons together
            ui.button('Dark', on_click=dark.enable).props("dense flat text-color=white")
            ui.button('Light', on_click=dark.disable).props("dense flat text-color=white")

    with ui.splitter(value=35).classes('w-full h-[calc(100vh-50px)] fixed top-[50px] left-0') as splitter:
        with splitter.before:
            initial_tab = app.storage.user.get('selected_tab_main', 'model')
            with ui.tabs().props('horizontal').classes('w-full').bind_value(app.storage.user, 'selected_tab_main', forward=lambda v: v or 'model') as main_tabs:
                model_tab = ui.tab('model',  icon='memory')
                model_tab.tooltip("Loads and shows all models")
                data_tab = ui.tab('data',  icon='table_chart')
                data_tab.tooltip("Loads and shows all datasets")
                edit_tab = ui.tab('edit',  icon='edit_note')
                edit_tab.tooltip("Loads and shows all .pt files and \n lets you edit the metadata or convert \n .pt files without type")
                merge_tab = ui.tab('merge', icon='merge_type') # New Merge tab
                merge_tab.tooltip("Loads and shows all datasets and let's you \n merge two datasets to a new one")

            with ui.tab_panels(main_tabs, value=initial_tab).props('vertical').classes('w-full h-full'):
                with ui.tab_panel(model_tab).classes('p-0'):
                    model_files_container = ui.refreshable(lambda: view_specific_file_list("model"))
                    create_main_list_content_for_view("model")
                with ui.tab_panel(data_tab).classes('p-0'):
                    data_files_container = ui.refreshable(lambda: view_specific_file_list("data"))
                    create_main_list_content_for_view("data")
                with ui.tab_panel(edit_tab).classes('p-0'):
                    edit_files_container = ui.refreshable(lambda: view_specific_file_list("edit"))
                    create_main_list_content_for_view("edit")
                with ui.tab_panel(merge_tab).classes('p-0'): # Panel for Merge view
                    merge_files_container = ui.refreshable(lambda: view_specific_file_list("merge"))
                    create_main_list_content_for_view("merge")
        
        with splitter.after:
            with ui.column().classes('w-full h-full p-0'):
                details_panel_content_area = ui.refreshable(render_details_panel_content)
                details_panel_content_area()

    # Initial scan for the default or stored tab's view if folder is set
    # Ensures content loads on page refresh if a folder was previously selected.
    await client.connected() # Wait for client to be connected
    active_view_on_load = app.storage.user.get('selected_tab_main', 'model')
    if APP_STATE[active_view_on_load]["folder"]:
         # Ensure SELECTED_FILE_CONTEXT is aligned with the loaded tab.
        if active_view_on_load == "merge":
            SELECTED_FILE_CONTEXT["view_key"] = "merge"
            SELECTED_FILE_CONTEXT["file_path"] = None
        else:
            # For other views, if a file was selected, its path might be in APP_STATE.
            # However, scan_folder_for_view will handle selection state.
            # We just need to ensure the context is not pointing to an old view's single file if merge tab is active.
            if SELECTED_FILE_CONTEXT["file_path"] and active_view_on_load != SELECTED_FILE_CONTEXT["view_key"]:
                 clear_selected_file_context_and_refresh_details()

        await scan_folder_for_view(active_view_on_load)
        details_panel_content_area.refresh() # Refresh right panel based on loaded tab

# --- Run the app ---
ui.run(title="Sphere Packing - Result Viewer", storage_secret="my_super_secret_key_12345", dark=None, reload=True)