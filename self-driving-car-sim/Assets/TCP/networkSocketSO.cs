using UnityEngine;
using System.Collections;
using System;
using System.IO;
using System.Net.Sockets;
using Newtonsoft.Json;
using System.Threading;

public class networkSocketSO : MonoBehaviour
{
	public String host = "localhost";
	public Int32 port = 50000;

	internal Boolean socket_ready = false;
	internal String input_buffer = "";

	TcpClient tcp_socket;
	NetworkStream net_stream;

	StreamWriter socket_writer;
	StreamReader socket_reader;

	private void Start()
	{
		setupSocket();
	}

	void Update()
	{
		string received_data = readSocket();
		switch (received_data)
		{
		case "pong":
			Debug.Log("Python controller sent: " + (string)received_data);
			writeSocket("ping");
			break;
		default:
			Debug.Log("Nothing received");
			break;
		}
	}

	void OnApplicationQuit()
	{
		closeSocket();
	}

	// Helper methods for:
	//...setting up the communication
	public void setupSocket()
	{
		try
		{
			tcp_socket = new TcpClient(host, port);
			net_stream = tcp_socket.GetStream();
			socket_writer = new StreamWriter(net_stream);
			socket_reader = new StreamReader(net_stream);
			socket_ready = true;
		}
		catch (Exception e)
		{
			// Something went wrong
			Debug.Log("Socket error: " + e);
		}
	}

	//... writing to a socket...
	public void writeSocket(string line)
	{
		if (!socket_ready)
			return;

		line = line + "\r\n";
		socket_writer.Write(line);
		socket_writer.Flush();
	}

	//... reading from a socket...
	public String readSocket()
	{
		if (!socket_ready)
			return "";

		if (net_stream.DataAvailable)
			return socket_reader.ReadLine();

		return "";
	}

	//... closing a socket...
	public void closeSocket()
	{
		if (!socket_ready)
			return;

		socket_writer.Close();
		socket_reader.Close();
		tcp_socket.Close();
		socket_ready = false;
	}
}