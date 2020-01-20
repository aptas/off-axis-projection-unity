using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.IO;
using UnityEngine;

namespace Apt.Unity.Projection
{
    // TCP code: https://gist.github.com/danielbierwirth/0636650b005834204cb19ef5ae6ccedb
    // JSON Unity: https://github.com/tawnkramer/sdsandbox/tree/master &
    // https://assetstore.unity.com/packages/tools/input-management/json-object-710 &
    // https://github.com/mtschoen/JSONObject

    public class AutoMoveTracker : TrackerBase
    {
        [Min(0.001f)]
        public float BoundsSize = 2;

        [Range(0, 1)]
        public float XMovement = 0.5f;
        [Range(0, 1)]
        public float YMovement = 0.3f;
        [Range(0, 1)]
        public float ZMovement = 0;

        private float HalfBoundSize => BoundsSize * 0.5f;

        #region private members 	
        /// <summary> 	
        /// TCPListener to listen for incomming TCP connection 	
        /// requests. 	
        /// </summary> 	
        private TcpListener tcpListener;
        /// <summary> 
        /// Background thread for TcpServer workload. 	
        /// </summary> 	
        private Thread tcpListenerThread;
        /// <summary> 	
        /// Create handle to connected tcp client. 	
        /// </summary> 	
        private TcpClient connectedTcpClient;

        private JSONObject irisesJson;

        private Irises irises = new Irises();

        private bool firstmsg = true;
        private float zoffset = 0.0f;

        //private offset = new 
        #endregion

        #region public members
        public float speedH = 2.0f;
        public float speedV = 2.0f;
        public float speedZ = 2.0f;

        public float screenHeight_cms = 16;
        public float scaling;
        #endregion

        int counter = 0;

        void Start()
        {
            IsTracking = false;

            scaling = 4.5f / screenHeight_cms; //screen height 3d world/height in cm

            // Start TcpServer background thread 		
            tcpListenerThread = new Thread(new ThreadStart(ListenForIncommingRequests));
            tcpListenerThread.IsBackground = true;
            tcpListenerThread.Start();
        }

        void Update()
        {
            if (Input.GetKeyUp(KeyCode.S) || Input.GetKey(KeyCode.KeypadEnter))
            {
                IsTracking = !IsTracking;
                SecondsHasBeenTracked = 0;
            }

            if (Input.GetKeyUp(KeyCode.Escape))
            {
                Application.Quit();
            }

            XMovement = irises.eyeX;
            YMovement = irises.eyeY;
            ZMovement = irises.eyeZ;

            if(IsTracking)
            {
                //SecondsHasBeenTracked += Time.deltaTime;
                //float xSize = XMovement * HalfBoundSize * scaling;
                //float ySize = YMovement * HalfBoundSize * scaling;
                //float zSize = ZMovement * HalfBoundSize * scaling;
                //translation.x =  Mathf.Sin(SecondsHasBeenTracked) * xSize;
                //translation.y =  Mathf.Sin(SecondsHasBeenTracked - (Mathf.PI * 2 / 3)) * ySize;
                //translation.z =  Mathf.Sin(SecondsHasBeenTracked) * zSize;
                translation.x = XMovement * scaling;
                translation.y = YMovement * scaling;
                translation.z = -ZMovement * scaling;
            }

        }

        /// <summary> 	
        /// Runs in background TcpServerThread; Handles incomming TcpClient requests 	
        /// </summary> 	
        private void ListenForIncommingRequests()
        {
            try
            {
                // Create listener on localhost port 8052. 			
                tcpListener = new TcpListener(IPAddress.Parse("127.0.0.1"), 8080);
                tcpListener.Start();
                Debug.Log("Server is listening");
                Byte[] bytes = new Byte[1024];
                while (true)
                {
                    using (connectedTcpClient = tcpListener.AcceptTcpClient())
                    {
                        // Get a stream object for reading 					
                        using (NetworkStream stream = connectedTcpClient.GetStream())
                        {
                            int length;
                            // Read incomming stream into byte arrary. 						
                            while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
                            {
                                var incommingData = new byte[length];
                                Array.Copy(bytes, 0, incommingData, 0, length);
                                // Convert byte array to JSON message. 							
                                String clientMessage = Encoding.UTF8.GetString(incommingData);
                                // Added: convert string to JSON
                                JSONObject clientMessage_json = new JSONObject(clientMessage);
                                //Debug.Log("client message received as: " + clientMessage_json);
                                irisesJson = clientMessage_json.Copy();
                                ReadIrises(clientMessage_json);
                            }
                        }
                    }
                }
            }
            catch (SocketException socketException)
            {
                Debug.Log("SocketException " + socketException.ToString());
            }
        }

        // Use JSON message to read irises positions
        public void ReadIrises(JSONObject clientMessage_json)
        {
            irises = new Irises(clientMessage_json["eyeX"].ToString(),
                                clientMessage_json["eyeY"].ToString(),
                                clientMessage_json["eyeZ"].ToString());
            if (firstmsg)
            {
                zoffset = irises.eyeZ;
                firstmsg = false;
                IsTracking = true;
            }
            // Tell the python client we received the message
            SendMessage();
        }

        /// <summary> 	
        /// Send message to client using socket connection. 	
        /// </summary> 	
        private void SendMessage()
        {
            if (connectedTcpClient == null)
            {
                return;
            }

            try
            {
                // Get a stream object for writing. 			
                NetworkStream stream = connectedTcpClient.GetStream();
                if (stream.CanWrite)
                {
                    // Added: create dict to be a JSON object
                    Dictionary<string, string> serverMessage = new Dictionary<string, string>();
                    serverMessage["unity"] = String.Format("Unity sends its regards {0}", counter);
                    counter++;
                    JSONObject serverMessage_json = new JSONObject(serverMessage);
                    String serverMessage_string = serverMessage_json.ToString();
                    //string serverMessage = "This is a message from your server.";  // original code			
                    // Convert string message to byte array.                 
                    byte[] serverMessageAsByteArray = Encoding.UTF8.GetBytes(serverMessage_string);  // serverMessage				
                                                                                                     // Write byte array to socketConnection stream.               
                    stream.Write(serverMessageAsByteArray, 0, serverMessageAsByteArray.Length);
                    //Debug.Log("Server sent his message - should be received by client");
                }
            }
            catch (SocketException socketException)
            {
                Debug.Log("Socket exception: " + socketException);
            }
        }
    }

    public class Irises
    {
        public float eyeX = 0;
        public float eyeY = 0;
        public float eyeZ = 0;

        public Irises()
        { }
        public Irises(string sx, string sy, string sz)
        {
            eyeX = float.Parse(sx);
            eyeY = float.Parse(sy);
            eyeZ = float.Parse(sz);
        }
    }
}