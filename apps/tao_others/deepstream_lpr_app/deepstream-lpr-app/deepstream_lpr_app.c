/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gmodule.h>
#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"
#include "nvdsmeta.h"
#include <gst/rtsp-server/rtsp-server.h>
#include "nvds_yml_parser.h"
#include "ds_yml_parse.h"

#define MAX_DISPLAY_LEN 64

#define MEASURE_ENABLE 1

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define SGIE_CLASS_ID_LPD 0 

/* Enhanced muxer output resolution - can be modified as needed */
#define MUXER_OUTPUT_WIDTH 1920   // Changed from 1280 to 1920 (1080p)
#define MUXER_OUTPUT_HEIGHT 1080  // Changed from 720 to 1080 (1080p)

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

/* Frame rate control - 30 FPS */
#define TARGET_FPS 30
#define FRAME_DURATION_NS (1000000000 / TARGET_FPS)

#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GPU_ID "gpu-id"
#define TRITON  "triton"
#define NVINFER_LPD_CH_CFG "../../../configs/nvinfer/LPD_us_tao/sgie_lpd_DetectNet2_us.txt"
#define NVINFER_LDP_US_CFG "../../../configs/nvinfer/LPD_ch_tao/sgie_lpd_DetectNet2_ch.txt"

gint frame_number = 0;
gint total_plate_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

extern "C" void
parse_nvdsanalytics_meta_data (NvDsBatchMeta *batch_meta);

#define PRIMARY_DETECTOR_UID 1
#define SECONDARY_DETECTOR_UID 2
#define SECONDARY_CLASSIFIER_UID 3

typedef struct _perf_measure{
  GstClockTime pre_time;
  GstClockTime total_time;
  guint count;
}perf_measure;

/* Enhanced streammux configuration function */
static void
configure_streammux_enhanced(GstElement *streammux, guint sources, 
                           guint width, guint height, guint fps)
{
  guint frame_duration_ns = 1000000000 / fps;  // Convert fps to nanoseconds
  
  g_object_set (G_OBJECT (streammux),
      "width", width,
      "height", height,
      "batch-size", sources,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC,
      
      // Enhanced frame processing properties
      "live-source", FALSE,                    // Set to TRUE for live sources
      "buffer-pool-size", 8,                   // Increased buffer pool
      "num-surfaces-per-frame", 1,             // Surfaces per frame
      "interpolation-method", 5,               // Advanced scaling algorithm (Algo1)
      "gpu-id", 0,                            // GPU device ID
      "nvbuf-memory-type", 0,                 // 0=Default, 1=CUDA_PINNED, 2=CUDA_DEVICE
      "sync-inputs", TRUE,                    // Synchronize input streams
      "max-latency", 200000000,               // Max latency in nanoseconds (200ms)
      "frame-duration", frame_duration_ns,     // Dynamic frame duration
      "drop-pipeline-eos", FALSE,             // Don't drop EOS events
      "compute-hw", 1,                        // Use GPU for processing
      "enable-padding", TRUE,                 // Enable padding for aspect ratio
      "frame-num-reset-on-eos", TRUE,         // Reset frame numbers on EOS
      "frame-num-reset-on-stream-reset", TRUE, // Reset on stream reset
      NULL);
  
  g_print("Streammux configured: %dx%d @ %d FPS with %d sources\n", 
          width, height, fps, sources);
}

/* Function to dynamically update streammux properties */
static void
update_streammux_resolution(GstElement *streammux, guint new_width, guint new_height)
{
  g_object_set (G_OBJECT (streammux),
      "width", new_width,
      "height", new_height,
      NULL);
  g_print("Streammux resolution updated to: %dx%d\n", new_width, new_height);
}

static gchar *
get_absolute_file_path (gchar *cfg_file_path, gchar *file_path)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path && file_path[0] == '/') {
    return file_path;
  }

  if (!realpath (cfg_file_path, abs_cfg_path)) {
    g_free (file_path);
    return NULL;
  }

  /* Return absolute path of config file if file_path is NULL. */
  if (!file_path) {
    abs_file_path = g_strdup (abs_cfg_path);
    return abs_file_path;
  }

  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
  g_free (file_path);

  return abs_file_path;
}

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  guint lp_count = 0;
  guint label_i = 0;
  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_obj = NULL;
  NvDsMetaList * l_class = NULL;
  NvDsMetaList * l_label = NULL;
  NvDsDisplayMeta *display_meta = NULL;
  NvDsClassifierMeta * class_meta = NULL;
  NvDsLabelInfo * label_info = NULL;
  GstClockTime now;
  perf_measure * perf = (perf_measure *)(u_data);

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  now = g_get_monotonic_time();

  if (perf->pre_time == GST_CLOCK_TIME_NONE) {
    perf->pre_time = now;
    perf->total_time = GST_CLOCK_TIME_NONE;
  } else {
    if (perf->total_time == GST_CLOCK_TIME_NONE) {
      perf->total_time = (now - perf->pre_time);
    } else {
      perf->total_time += (now - perf->pre_time);
    }
    perf->pre_time = now;
    perf->count++;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    int offset = 0;
    if (!frame_meta)
      continue;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);

      if (!obj_meta)
        continue;

      /* Check that the object has been detected by the primary detector
      * and that the class id is that of vehicles/persons. */
      if (obj_meta->unique_component_id == PRIMARY_DETECTOR_UID) {
        if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
          vehicle_count++;
        if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
          person_count++;
      }

      if (obj_meta->unique_component_id == SECONDARY_DETECTOR_UID) {
        if (obj_meta->class_id == SGIE_CLASS_ID_LPD) {
          lp_count++;
          /* Print this info only when operating in secondary model. */
          if (obj_meta->parent)
            g_print ("License plate found for parent object %p (type=%s)\n",
              obj_meta->parent, pgie_classes_str[obj_meta->parent->class_id]);

          obj_meta->text_params.set_bg_clr = 1;
          obj_meta->text_params.text_bg_clr.red = 0.0;
          obj_meta->text_params.text_bg_clr.green = 0.0;
          obj_meta->text_params.text_bg_clr.blue = 0.0;
          obj_meta->text_params.text_bg_clr.alpha = 0.0;

          obj_meta->text_params.font_params.font_color.red = 1.0;
          obj_meta->text_params.font_params.font_color.green = 1.0;
          obj_meta->text_params.font_params.font_color.blue = 0.0;
          obj_meta->text_params.font_params.font_color.alpha = 1.0;
          obj_meta->text_params.font_params.font_size = 12;
        }
      }

      for (l_class = obj_meta->classifier_meta_list; l_class != NULL;
           l_class = l_class->next) {
        class_meta = (NvDsClassifierMeta *)(l_class->data);
        if (!class_meta)
          continue;
        if (class_meta->unique_component_id == SECONDARY_CLASSIFIER_UID) {
          for ( label_i = 0, l_label = class_meta->label_info_list;
            label_i < class_meta->num_labels && l_label; label_i++,
            l_label = l_label->next) {
              label_info = (NvDsLabelInfo *)(l_label->data);
              if (label_info) {
                if (label_info->label_id == 0 && label_info->result_class_id == 1) {
                  g_print ("Plate License %s\n",label_info->result_label);
                }
              }
          }
        }
      }
    }

    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char*) g_malloc0 (MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN,
                 "Person = %d ", person_count);
    offset += snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN,
                 "Vehicle = %d ", vehicle_count);

    /* Now set the offsets where the string should appear */
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    /* Font , font-color and font-size */
    char font_n[6];
    snprintf(font_n, 6, "Serif");
    txt_params->font_params.font_name = font_n;
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    /* Text background color */
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }

  g_print ("Frame Number = %d Vehicle Count = %d Person Count = %d"
           " License Plate Count = %d\n",
           frame_number, vehicle_count, person_count, lp_count);
  frame_number++;
  total_plate_number += lp_count;
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_new_pad (GstElement *element,
        GstPad     *pad,
        GstElement *data)
{
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;
  GstPadLinkReturn ret;

  GstPad *sink_pad = gst_element_get_static_pad (data, "sink");
  if (gst_pad_is_linked (sink_pad)) {
    g_print ("h264parser already linked. Ignoring.\n");
    goto exit;
  }

  new_pad_caps = gst_pad_get_current_caps (pad);
  new_pad_struct = gst_caps_get_structure (new_pad_caps, 0);
  new_pad_type = gst_structure_get_name (new_pad_struct);
  g_print("qtdemux pad %s\n",new_pad_type);
  
  if (g_str_has_prefix (new_pad_type, "video/x-h264")) {
    ret = gst_pad_link (pad, sink_pad);
    if(GST_PAD_LINK_FAILED (ret))
      g_print ("fail to link parser and mp4 demux.\n");
  } else {
    g_print ("%s output, not 264 stream\n",new_pad_type);
  }

exit:
  gst_object_unref (sink_pad);
}

/* Tracker config parsing */

#define CHECK_ERROR(error) \
  if (error) { \
    g_printerr ("Error while parsing config file: %s\n", error->message); \
    goto done; \
  }

static gboolean
set_tracker_properties (GstElement *nvtracker, char * config_file_name)
{
  gboolean ret = FALSE;
  GError *error = NULL;
  gchar **keys = NULL;
  gchar **key = NULL;
  GKeyFile *key_file = g_key_file_new ();

  if (!g_key_file_load_from_file (key_file, config_file_name, G_KEY_FILE_NONE,
                                  &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    return FALSE;
  }

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TRACKER, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_WIDTH)) {
      gint width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_WIDTH, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-width", width, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
      gint height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_HEIGHT, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-height", height, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      guint gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "gpu_id", gpu_id, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
      char* ll_config_file = get_absolute_file_path (config_file_name,
                    g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-config-file",
          ll_config_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
      char* ll_lib_file = get_absolute_file_path (config_file_name,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-lib-file", ll_lib_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
      gboolean enable_batch_process =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "enable_batch_process",
                    enable_batch_process, NULL);
    } else {
      g_printerr ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TRACKER);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    g_printerr ("%s failed", __func__);
  }
  return ret;
}

/* nvdsanalytics_src_pad_buffer_probe  will extract metadata received on
 * nvdsanalytics src pad and extract nvanalytics metadata etc. */
static GstPadProbeReturn
nvdsanalytics_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  parse_nvdsanalytics_meta_data (batch_meta);

  return GST_PAD_PROBE_OK;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL,*streammux = NULL, *sink = NULL, 
             *primary_detector = NULL, *secondary_detector = NULL,
             *nvvidconv = NULL, *nvosd = NULL, *nvvidconv1 = NULL,
             *outenc = NULL, *capfilt = NULL, *mux = NULL,
             *secondary_classifier = NULL, *nvtile=NULL, *encparse = NULL;
  GstElement *tracker = NULL, *nvdsanalytics = NULL;
  GstElement *queue1 = NULL, *queue2 = NULL, *queue3 = NULL, *queue4 = NULL,
             *queue5 = NULL, *queue6 = NULL, *queue7 = NULL, *queue8 = NULL,
             *queue9 = NULL, *queue10 = NULL;
  GstElement *h264parser[128], *source[128], *decoder[128], *mp4demux[128],
             *parsequeue[128];
  char pgie_cfg_file_path[256] = {0};
  char lpd_cfg_file_path[256] = {0};
  char lpr_cfg_file_path[256] = {0};
#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  //int i;
  static guint src_cnt = 0;
  guint tiler_rows, tiler_columns;
  perf_measure perf_measure;

  gchar ele_name[64];
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  bool isYAML=false;
  bool isH264=true;
  int enc_type = ENCODER_TYPE_HW;
  GList* g_list = NULL;
  GList* iterator = NULL;
    
  const char*  ptriton = "triton";
  const char*  ptriton_grpc = "tritongrpc";
  const char *infer_plugin = "nvinfer";
  gboolean use_nvinfer_server = false;
  gboolean use_triton_grpc = false;
  guint car_mode = 1;
  
  /* Check input arguments */
  if (argc == 2 && (g_str_has_suffix(argv[1], ".yml")
      || (g_str_has_suffix(argv[1], ".yaml")))) {
    isYAML=TRUE;
  } else if (argc < 6 || argc > 133 || (atoi(argv[1]) != 1 && atoi(argv[1]) != 2) ||
     (atoi(argv[2]) != 1 && atoi(argv[2]) != 2 && atoi(argv[2]) != 3) ||
     (atoi(argv[3]) != 1 && atoi(argv[3]) != 0) ||
     (strcmp("infer", argv[4]) && strcmp(ptriton, argv[4])
      && strcmp(ptriton_grpc, argv[4]))) {
    g_printerr ("Usage: %s [1:us model|2: ch_model] [1:file sink|2:fakesink|"
        "3:display sink] [0:ROI disable|1:ROI enable] [infer|triton|tritongrpc] <In mp4 filename> <in mp4 filename> ... "
        "<out H264 filename>\n", argv[0]);
    g_printerr ("Enhanced version with improved frame processing capabilities\n");
    return -1;
  }

  //For Chinese language supporting
  setlocale(LC_CTYPE, "");
  if (isYAML) {
    if(ds_parse_group_enable(argv[1], TRITON)){
      use_nvinfer_server = true;
      infer_plugin = "nvinferserver";

      if(ds_parse_group_type(argv[1], TRITON) == 1){
        use_triton_grpc = true;
      }
    }

  } else {
    if (!strcmp(ptriton, argv[4])|| !strcmp(ptriton_grpc, argv[4])) {
      use_nvinfer_server = true;
      infer_plugin = "nvinferserver";
      if(!strcmp(ptriton_grpc, argv[4])){
        use_triton_grpc = true;
      }
    }
  }
  g_print("use_nvinfer_server:%d, use_triton_grpc:%d\n", use_nvinfer_server, use_triton_grpc);
  
  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);
  
  perf_measure.pre_time = GST_CLOCK_TIME_NONE;
  perf_measure.total_time = GST_CLOCK_TIME_NONE;
  perf_measure.count = 0;  

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("pipeline"); 

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
      g_printerr ("One element could not be created. Exiting.\n");
      return -1;
  }

  gst_bin_add (GST_BIN(pipeline), streammux);

  if(!isYAML) {
    for (src_cnt=0; src_cnt<(guint)argc-6; src_cnt++) {
      g_list = g_list_append(g_list, argv[src_cnt + 5]);
    }
  } else {
      if (NVDS_YAML_PARSER_SUCCESS != nvds_parse_source_list(&g_list, argv[1], "source-list")) {
        g_printerr ("No source is found. Exiting.\n");
        return -1;
      }
  }
  
  /* Multiple source files */
  for (iterator = g_list, src_cnt=0; iterator; iterator = iterator->next,src_cnt++) {
    /* Only h264 element stream with mp4 container is supported. */
    g_snprintf (ele_name, 64, "file_src_%d", src_cnt);

    /* Source element for reading from the file */
    source[src_cnt] = gst_element_factory_make ("filesrc", ele_name);

    g_snprintf (ele_name, 64, "mp4demux_%d", src_cnt);
    mp4demux[src_cnt] = gst_element_factory_make ("qtdemux", ele_name);

    g_snprintf (ele_name, 64, "h264parser_%d", src_cnt);
    h264parser[src_cnt] = gst_element_factory_make ("h264parse", ele_name);
      
    g_snprintf (ele_name, 64, "parsequeue_%d", src_cnt);
    parsequeue[src_cnt] = gst_element_factory_make ("queue", ele_name);

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    g_snprintf (ele_name, 64, "decoder_%d", src_cnt);
    decoder[src_cnt] = gst_element_factory_make ("nvv4l2decoder", ele_name);
      
    if(!source[src_cnt] || !h264parser[src_cnt] || !decoder[src_cnt] ||
       !mp4demux[src_cnt]) {
      g_printerr ("One element could not be created. Exiting.\n");
      return -1;
    }
      
    gst_bin_add_many (GST_BIN (pipeline), source[src_cnt], mp4demux[src_cnt],
        h264parser[src_cnt], parsequeue[src_cnt], decoder[src_cnt], NULL);
      
    g_snprintf (pad_name_sink, 64, "sink_%d", src_cnt);
    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    g_print("Request %s pad from streammux\n",pad_
