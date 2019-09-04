#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"

static int DEBUG = 0;

int main(int argc, char** argv)
{
    char *cfgfile = argv[1];
    char *weightfile = argv[2];

    if (argc < 2)
    {
        printf("Usage: %s <cfgfile> <weightfile>\n", argv[0]);
        return 255;
    }

    srand(2222222);

    printf("Loading network");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    #ifdef NNPACK
        nnp_initialize();
        #ifdef QPU_GEMM
            net->threadpool = pthreadpool_create(1);
        #else
            net->threadpool = pthreadpool_create(4);
        #endif
    #endif

    CvCapture *cap = cvCaptureFromCAM(0);

    if (!cap)
    {
        printf("Couldnt initialize camera.");
        return 2;
    }

    cvNamedWindow("Main", CV_WINDOW_NORMAL);
    cvMoveWindow("Main", 0, 0);
    cvResizeWindow("Main", 800, 480);

    IplImage* ipl = 0;
    image img, letterImage;
    int nboxes = 0;
    detection *dets;

    float thresh = .5;
    float hier_thresh = .25;
    float nms = .4;

    int classToFollow = 0;

    img = get_image_from_stream(cap);
    letterImage = letterbox_image(img, net->w, net->h);

    for (;;)
    {
        ipl = cvCreateImage(cvSize(img.w, img.h), IPL_DEPTH_8U, img.c);

        layer l = net->layers[net->n-1];
        int status = fill_image_from_stream(cap, img);
        letterbox_image_into(img, net->w, net->h, letterImage);

        network_predict(net, letterImage.data);

        dets = get_network_boxes(net, img.w, img.h, thresh, hier_thresh, 0, 1, &nboxes);

        if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

        if (nboxes > 0)
        {
            show_image_cv(img, "Main", ipl);

            int i,j;

            for (i = 0; i < nboxes; ++i)
            {
                for(j = 0; j < l.classes; ++j)
                {
                    float prob = dets[i].prob[j]*100;

                    if (prob > 0)
                    {
                        if (DEBUG)
                            printf("Found class %i with prob %f \n",j,prob);
                    }

                    if (dets[i].prob[j] > thresh && j == classToFollow)
                    {
                        box b = dets[i].bbox;

                        float left  = (b.x-b.w/2.);
                        float right = (b.x+b.w/2.);
                        float top   = (b.y-b.h/2.);
                        float bot   = (b.y+b.h/2.);

                        float centerx = (right - left)/2.;
                        float centery = (bot - top)/2.;

                        if (DEBUG)
                            printf("Found box for class %i with coordinates (X,Y) = (%f,%f)\n\n",j, b.x, b.y);

                        char text[10];

                        if (b.x > .5)
                        {
                            sprintf(text, "%s", "RIGHT");
                        }
                        else
                        {
                            sprintf(text, "%s", "LEFT");
                        }

                        printf("%s\n", text);

                        CvFont font;
                        cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.3, 1.3, 0, 5, 8);
                        cvPutText(ipl, text, cvPoint(60, 60), &font, cvScalar(0,0,255,0));

                        cvShowImage("Main", ipl);

                    }
                }

            }

        }


        if( cvWaitKey(10) >= 0 )
            break;
    }

    cvReleaseCapture(&cap);
    cvDestroyWindow("Main");

    #ifdef NNPACK
        pthreadpool_destroy(net->threadpool);
        nnp_deinitialize();
    #endif

    free_network(net);

}
