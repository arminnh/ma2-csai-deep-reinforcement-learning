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
		touchesTrack = IsGrounded();
	}

	public bool IsGrounded()  
	{
		RaycastHit hit;
		if (Physics.Raycast (transform.position, -Vector3.up, out hit, 5)) {
			if (hit.collider != null && hit.collider.name == "GroundTrack")
				return true;
		}
		return false;
	}
		
}
