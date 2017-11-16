using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarTouchesTrack : MonoBehaviour {

	public bool touchesTrack;
	public bool test;

	// Use this for initialization
	void Start () {
		
	}
		
	void Update()
	{

	}

	void OnCollisionEnter(Collision collision)
	{
			touchesTrack = true;
	}

	void OnCollisionStay(Collision collision){
		Debug.Log (collision.gameObject.name);
	}

	void OnCollisionExit(Collision collision)
	{

		touchesTrack = false;
	}
}
